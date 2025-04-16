class Scheduler:
    """
    Scheduler for Gamma accelerator.
    
    Responsible for:
    1. Creating tasks from matrix A rows
    2. Building balanced, top-full task trees for rows with many nonzeros
    3. Dynamically scheduling tasks to PEs
    4. Managing partial output fibers
    5. Coordinating with FiberCache for data movement
    """
    def __init__(self, num_pes=32, pe_radix=64, pe_array=None, fiber_cache_manager=None, memory_controller=None):
        """
        Initialize the Scheduler.
        
        Args:
            num_pes: Number of Processing Elements in the system
            pe_radix: Radix of each PE (how many input fibers it can merge)
            pe_array: Reference to the PE array
            fiber_cache_manager: Reference to the FiberCache manager
            memory_controller: Reference to the memory controller
        """
        self.num_pes = num_pes
        self.pe_radix = pe_radix
        self.pe_array = pe_array
        self.fiber_cache = fiber_cache_manager
        self.memory_controller = memory_controller
        
        # Track PE states
        self.pe_busy = [False] * num_pes
        self.pe_current_task = [None] * num_pes
        self.pe_next_task = [None] * num_pes
        
        # Track task trees and dependencies
        self.task_trees = {}  # Maps row_id to task tree
        self.pending_tasks = []  # Tasks ready to be scheduled
        self.waiting_tasks = {}  # Tasks waiting for dependencies (task_id -> list of dependency task_ids)
        self.completed_tasks = set()  # Set of completed task IDs
        
        # Memory management for partial outputs
        self.partial_output_address = {}  # Maps partial output ID to memory address
        self.partial_output_size = {}  # Maps partial output ID to estimated size
        self.next_partial_output_address = 0  # Simple address allocation (would be more complex in real implementation)
        self.memory_regions = {}  # Track memory regions allocated for matrices
        
        # Task ID generator
        self.next_task_id = 0
        
        # Current cycle
        self.current_cycle = 0
        
        # Statistics
        self.stats = {
            'total_tasks_created': 0,
            'total_tasks_completed': 0,
            'pe_utilization': [0] * num_pes,
            'cycles': 0,
            'total_pe_busy_cycles': 0,
            'avg_task_latency': 0,
            'total_task_latency': 0,
            'max_pending_tasks': 0,
            'fibers_prefetched': 0,
            'memory_allocated': 0,
        }
    
    def load_matrices(self, matrix_A, matrix_B):
        """
        Load matrices A and B into memory and register with FiberCache.
        
        Args:
            matrix_A: CSR representation of input matrix A (offsets, coords, values)
            matrix_B: CSR representation of input matrix B (offsets, coords, values)
            
        Returns:
            Boolean indicating success
        """
        # Allocate memory regions for matrices
        # In a real implementation, this would use actual memory allocation
        a_size = sum(len(v) for v in matrix_A)
        b_size = sum(len(v) for v in matrix_B)
        
        a_address = self.next_partial_output_address
        self.next_partial_output_address += a_size
        
        b_address = self.next_partial_output_address
        self.next_partial_output_address += b_size
        
        self.memory_regions['A'] = (a_address, a_size)
        self.memory_regions['B'] = (b_address, b_size)
        
        self.stats['memory_allocated'] += a_size + b_size
        
        # Register matrix B fibers with FiberCache for future access
        offsets_b, coords_b, values_b = matrix_B
        num_rows_b = len(offsets_b) - 1
        
        for row_id in range(num_rows_b):
            start = offsets_b[row_id]
            end = offsets_b[row_id + 1]
            
            # Calculate row address based on the memory region and offsets
            row_address = b_address + start
            row_size = (end - start) * 12  # Approx size for coordinate-value pairs
            
            # Register fiber with FiberCache
            if self.fiber_cache:
                self.fiber_cache.register_fiber(f"B_{row_id}", row_address, row_size)
        
        return True
    
    def process_matrix_A(self, matrix_A):
        """
        Process matrix A to generate tasks.
        
        Args:
            matrix_A: CSR representation of input matrix A (offsets, coords, values)
            
        Returns:
            Number of tasks created
        """
        offsets, coords, values = matrix_A
        num_rows = len(offsets) - 1
        total_tasks = 0
        
        for row_id in range(num_rows):
            start = offsets[row_id]
            end = offsets[row_id + 1]
            nonzeros = end - start
            
            if nonzeros == 0:
                # Skip empty rows
                continue
            
            if nonzeros <= self.pe_radix:
                # Simple case: row fits in a single PE
                task_id = self._get_next_task_id()
                
                # Create a task to process this row directly
                task = {
                    'id': task_id,
                    'type': 'direct',
                    'row_id': row_id,
                    'A_coords': coords[start:end].tolist() if hasattr(coords, 'tolist') else list(coords[start:end]),
                    'A_values': values[start:end].tolist() if hasattr(values, 'tolist') else list(values[start:end]),
                    'B_row_ids': coords[start:end].tolist() if hasattr(coords, 'tolist') else list(coords[start:end]),
                    'scaling_factors': values[start:end].tolist() if hasattr(values, 'tolist') else list(values[start:end]),
                    'output_id': f"C_{row_id}",
                    'dependencies': [],
                    'created_cycle': self.current_cycle
                }
                
                # Pre-register output fiber
                self._allocate_output(f"C_{row_id}", nonzeros * 12)
                
                self.pending_tasks.append(task)
                total_tasks += 1
            else:
                # Complex case: need to build a task tree
                tree = self._build_task_tree(row_id, coords[start:end], values[start:end])
                self.task_trees[row_id] = tree
                total_tasks += len(tree['tasks'])
                
                # Add leaf tasks to pending tasks
                for task in tree['leaf_tasks']:
                    self.pending_tasks.append(task)
                
                # Add non-leaf tasks to waiting tasks
                for task_id, dependencies in tree['dependencies'].items():
                    self.waiting_tasks[task_id] = dependencies
        
        self.stats['total_tasks_created'] += total_tasks
        self.stats['max_pending_tasks'] = max(self.stats['max_pending_tasks'], len(self.pending_tasks))
        
        return total_tasks
    
    def _build_task_tree(self, row_id, coords, values):
        """
        Build a balanced, top-full task tree for a row with many nonzeros.
        
        Args:
            row_id: Row ID in matrix A
            coords: Column coordinates (K dimension) for this row
            values: Values for this row
            
        Returns:
            Task tree structure
        """
        # Convert numpy arrays to lists if needed
        if hasattr(coords, 'tolist'):
            coords = coords.tolist()
        if hasattr(values, 'tolist'):
            values = values.tolist()
        
        # Determine number of levels needed
        nonzeros = len(coords)
        num_leaf_tasks = (nonzeros + self.pe_radix - 1) // self.pe_radix
        
        # Ensure we build a balanced, top-full tree
        levels = 0
        while (1 << levels) < num_leaf_tasks:
            levels += 1
        
        # Create task tree structure
        tasks = []
        task_ids = []
        dependencies = {}  # task_id -> list of dependency task_ids
        leaf_tasks = []
        
        # Create the leaf tasks first
        for i in range(num_leaf_tasks):
            start = i * self.pe_radix
            end = min(start + self.pe_radix, nonzeros)
            
            task_id = self._get_next_task_id()
            task_ids.append(task_id)
            
            # Create partial output ID
            partial_output_id = f"partial_{row_id}_{i}"
            
            # Allocate memory for partial output
            estimated_size = (end - start) * 12  # Simple estimation
            self._allocate_partial_output(partial_output_id, estimated_size)
            
            # Create task
            task = {
                'id': task_id,
                'type': 'leaf',
                'row_id': row_id,
                'A_coords': coords[start:end],
                'A_values': values[start:end],
                'B_row_ids': coords[start:end],
                'scaling_factors': values[start:end],
                'output_id': partial_output_id,
                'dependencies': [],
                'created_cycle': self.current_cycle
            }
            
            tasks.append(task)
            leaf_tasks.append(task)
        
        # Create intermediate and root tasks
        level_start = 0
        level_size = num_leaf_tasks
        
        for level in range(levels):
            next_level_size = (level_size + 1) // 2
            next_level_start = level_start + level_size
            
            for i in range(next_level_size):
                child1_idx = level_start + i * 2
                child2_idx = child1_idx + 1 if i * 2 + 1 < level_size else None
                
                task_id = self._get_next_task_id()
                task_ids.append(task_id)
                
                # Create partial output ID (or final output for root)
                is_root = (level == levels - 1)
                if is_root:
                    output_id = f"C_{row_id}"  # Final output row
                    estimated_size = nonzeros * 12  # Simple estimation
                    self._allocate_output(output_id, estimated_size)
                else:
                    output_id = f"partial_{row_id}_{next_level_start + i}"
                    # Allocate memory for partial output
                    estimated_size = nonzeros * 8  # Simple estimation
                    self._allocate_partial_output(output_id, estimated_size)
                
                # Determine dependencies
                task_dependencies = [task_ids[child1_idx]]
                input_ids = [tasks[child1_idx]['output_id']]
                
                if child2_idx is not None:
                    task_dependencies.append(task_ids[child2_idx])
                    input_ids.append(tasks[child2_idx]['output_id'])
                
                # Create task
                task = {
                    'id': task_id,
                    'type': 'intermediate' if not is_root else 'root',
                    'row_id': row_id,
                    'input_ids': input_ids,
                    'output_id': output_id,
                    'dependencies': task_dependencies,
                    'created_cycle': self.current_cycle
                }
                
                tasks.append(task)
                dependencies[task_id] = task_dependencies
            
            level_start = next_level_start
            level_size = next_level_size
        
        return {
            'row_id': row_id,
            'tasks': tasks,
            'task_ids': task_ids,
            'dependencies': dependencies,
            'leaf_tasks': leaf_tasks,
            'root_task_id': task_ids[-1]
        }
    
    def _get_next_task_id(self):
        """
        Generate a unique task ID.
        
        Returns:
            Unique task ID
        """
        task_id = self.next_task_id
        self.next_task_id += 1
        return task_id
    
    def _allocate_partial_output(self, partial_output_id, estimated_size):
        """
        Allocate memory for a partial output fiber.
        
        Args:
            partial_output_id: ID of the partial output
            estimated_size: Estimated size in bytes
            
        Returns:
            Allocated address
        """
        # Simple allocation strategy - in a real implementation would be more complex
        address = self.next_partial_output_address
        self.next_partial_output_address += estimated_size
        
        self.partial_output_address[partial_output_id] = address
        self.partial_output_size[partial_output_id] = estimated_size
        
        # Register with FiberCache
        if self.fiber_cache:
            self.fiber_cache.register_fiber(partial_output_id, address, estimated_size)
        
        self.stats['memory_allocated'] += estimated_size
        
        return address
    
    def _allocate_output(self, output_id, estimated_size):
        """
        Allocate memory for a final output fiber.
        
        Args:
            output_id: ID of the output
            estimated_size: Estimated size in bytes
            
        Returns:
            Allocated address
        """
        # Simple allocation strategy - in a real implementation would be more complex
        address = self.next_partial_output_address
        self.next_partial_output_address += estimated_size
        
        # Register with FiberCache
        if self.fiber_cache:
            self.fiber_cache.register_fiber(output_id, address, estimated_size)
        
        self.stats['memory_allocated'] += estimated_size
        
        return address
    
    def _deallocate_partial_output(self, partial_output_id):
        """
        Deallocate memory for a partial output fiber.
        
        Args:
            partial_output_id: ID of the partial output
            
        Returns:
            Boolean indicating success
        """
        if partial_output_id not in self.partial_output_address:
            return False
        
        # In a real implementation, this would return memory to a pool
        address = self.partial_output_address[partial_output_id]
        size = self.partial_output_size[partial_output_id]
        
        del self.partial_output_address[partial_output_id]
        del self.partial_output_size[partial_output_id]
        
        self.stats['memory_allocated'] -= size
        
        return True
    
    def prefetch_fibers_for_task(self, task):
        """
        Prefetch fibers needed for a task.
        
        Args:
            task: Task to prefetch fibers for
            
        Returns:
            Number of fibers prefetched
        """
        prefetched = 0
        
        if task['type'] in ['direct', 'leaf']:
            # Prefetch B rows
            for b_row_id in task['B_row_ids']:
                if self.fiber_cache:
                    self.fiber_cache.fetch_fiber(f"B_{b_row_id}", None)
                    prefetched += 1
        else:
            # Prefetch partial outputs
            for input_id in task['input_ids']:
                if self.fiber_cache:
                    self.fiber_cache.fetch_fiber(input_id, None)
                    prefetched += 1
        
        self.stats['fibers_prefetched'] += prefetched
        return prefetched
    
    def schedule_tasks(self):
        """
        Schedule pending tasks to available PEs.
        
        Returns:
            Number of tasks scheduled
        """
        tasks_scheduled = 0
        
        # Update PE status
        if self.pe_array:
            for pe_id in range(self.num_pes):
                status = self.pe_array.get_pe_status(pe_id)
                if status:
                    self.pe_busy[pe_id] = status['busy']
        
        # Find idle PEs
        for pe_id in range(self.num_pes):
            if not self.pe_busy[pe_id] and self.pe_next_task[pe_id] is None:
                if self.pending_tasks:
                    # Get the next task
                    task = self.pending_tasks.pop(0)
                    
                    # Prefetch fibers for this task
                    self.prefetch_fibers_for_task(task)
                    
                    # Assign to PE
                    if self.pe_array:
                        self.pe_array.assign_task(pe_id, task)
                    
                    # Update state
                    self.pe_busy[pe_id] = True
                    self.pe_current_task[pe_id] = task
                    tasks_scheduled += 1
        
        self.stats['max_pending_tasks'] = max(self.stats['max_pending_tasks'], len(self.pending_tasks))
        return tasks_scheduled
    
    def handle_completed_tasks(self):
        """
        Handle tasks that have been completed by PEs.
        
        Returns:
            Number of tasks handled
        """
        if not self.pe_array:
            return 0
        
        completed = self.pe_array.get_completed_tasks()
        tasks_handled = 0
        
        for pe_id, task_id in completed:
            if task_id is None:
                continue
                
            # Mark task as completed
            self.completed_tasks.add(task_id)
            self.stats['total_tasks_completed'] += 1
            self.pe_busy[pe_id] = False
            self.pe_current_task[pe_id] = None
            tasks_handled += 1
            
            # Find the task in the task trees
            task = None
            task_row_id = None
            for row_id, tree in self.task_trees.items():
                for t in tree['tasks']:
                    if t['id'] == task_id:
                        task = t
                        task_row_id = row_id
                        break
                if task:
                    break
            
            if not task:
                # Task not found in task trees - might be a direct task
                continue
            
            # For intermediate tasks, we can deallocate consumed partial outputs
            if task['type'] == 'intermediate' or task['type'] == 'root':
                for input_id in task['input_ids']:
                    self._deallocate_partial_output(input_id)
            
            # Update dependencies for waiting tasks
            tasks_to_schedule = []
            
            for waiting_task_id, dependencies in list(self.waiting_tasks.items()):
                if task_id in dependencies:
                    dependencies.remove(task_id)
                    
                    if not dependencies:
                        # All dependencies satisfied - move to pending
                        for t in self.task_trees.get(task_row_id, {}).get('tasks', []):
                            if t['id'] == waiting_task_id:
                                tasks_to_schedule.append(t)
                                del self.waiting_tasks[waiting_task_id]
                                break
            
            # Add newly ready tasks to pending queue
            self.pending_tasks.extend(tasks_to_schedule)
            self.stats['max_pending_tasks'] = max(self.stats['max_pending_tasks'], len(self.pending_tasks))
        
        return tasks_handled
    
    def tick(self):
        """
        Advance the scheduler by one cycle.
        
        Returns:
            Status dictionary
        """
        self.current_cycle += 1
        self.stats['cycles'] = self.current_cycle
        
        # First handle any completed tasks
        completed_tasks = self.handle_completed_tasks()
        
        # Then schedule new tasks
        scheduled_tasks = self.schedule_tasks()
        
        # Count busy PEs for utilization statistics
        busy_pes = sum(1 for busy in self.pe_busy if busy)
        self.stats['total_pe_busy_cycles'] += busy_pes
        self.stats['pe_utilization'] = [self.pe_busy[i] for i in range(self.num_pes)]
        
        return {
            'cycle': self.current_cycle,
            'busy_pes': busy_pes,
            'pe_utilization': busy_pes / self.num_pes,
            'pending_tasks': len(self.pending_tasks),
            'waiting_tasks': len(self.waiting_tasks),
            'completed_tasks': completed_tasks,
            'scheduled_tasks': scheduled_tasks
        }
    
    def is_processing_complete(self):
        """
        Check if all tasks have been completed.
        
        Returns:
            Boolean indicating if processing is complete
        """
        return (len(self.pending_tasks) == 0 and 
                len(self.waiting_tasks) == 0 and 
                all(not busy for busy in self.pe_busy))
    
    def get_stats(self):
        """
        Get scheduler statistics.
        
        Returns:
            Dictionary of statistics
        """
        # Calculate overall PE utilization
        if self.stats['cycles'] > 0:
            avg_utilization = self.stats['total_pe_busy_cycles'] / (self.stats['cycles'] * self.num_pes)
        else:
            avg_utilization = 0
        
        stats = dict(self.stats)
        stats['avg_pe_utilization'] = avg_utilization
        
        return stats