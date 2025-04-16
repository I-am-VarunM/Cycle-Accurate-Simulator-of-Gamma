class ProcessingElement:
    def __init__(self, pe_id, radix=64, pipeline_depth=4, fiber_cache=None):
        """
        Initialize a Processing Element with the given ID and merger radix.
        
        Args:
            pe_id: Unique identifier for this PE
            radix: The number of input fibers the PE can merge (default 64)
            pipeline_depth: Number of pipeline stages
            fiber_cache: Reference to the FiberCache
        """
        self.pe_id = pe_id
        self.radix = radix
        self.pipeline_depth = pipeline_depth
        self.fiber_cache = fiber_cache
        
        # Pipeline stages
        # Stage 0: Fiber fetch stage
        # Stage 1: Merge stage
        # Stage 2: Scale stage (multiplication)
        # Stage 3: Accumulate stage
        self.pipeline = [None] * pipeline_depth
        
        # Input state
        self.fiber_buffers = [[] for _ in range(radix)]  # Circular buffers for input fibers
        self.fiber_buffer_heads = [0] * radix  # Current position in each buffer
        self.scaling_factors = [0.0] * radix  # Scaling factors for each input fiber
        self.fiber_ids = [None] * radix  # IDs of the fibers in the buffers
        self.fiber_done = [True] * radix  # Flag to track when a fiber is fully processed
        self.fiber_fetched = [False] * radix  # Flag to track if a fiber has been fetched
        
        # Merger state
        self.merger_state = {
            'head_indexes': [0] * radix,  # Current position in each input fiber
            'current_output_coord': None  # Current coordinate being processed
        }
        
        # Accumulator state
        self.accumulator_value = 0.0
        self.accumulator_coord = None
        
        # Output state
        self.output_fiber = []  # Results
        self.output_fiber_id = None  # ID of the output fiber
        
        # Task state
        self.current_task = None
        self.busy = False
        self.stalled = False
        self.stall_reason = None
        self.cycle_counter = 0
        self.task_start_cycle = 0
        self.task_id = None
        
        # Statistics
        self.stats = {
            'cycles': 0,
            'busy_cycles': 0,
            'stall_cycles': 0,
            'elements_processed': 0,
            'multiplications': 0,
            'additions': 0,
            'merges': 0,
            'fiber_reads': 0,
            'fiber_fetches': 0,
            'tasks_completed': 0
        }
    
    def set_task(self, task):
        """
        Set a new task for the PE to process.
        
        Args:
            task: Dictionary containing task information:
                - 'id': Task ID
                - 'type': Task type (direct, leaf, intermediate, root)
                - 'row_id': Row ID in matrix A
                - For direct/leaf tasks:
                    - 'B_row_ids': IDs of B rows needed
                    - 'scaling_factors': Scaling factors for each B row
                - For intermediate/root tasks:
                    - 'input_ids': IDs of input fibers
                - 'output_id': ID for the output fiber
                
        Returns:
            Boolean indicating if task was accepted
        """
        if self.busy:
            return False
        
        self.current_task = task
        self.task_id = task['id']
        self.busy = True
        self.stalled = False
        self.stall_reason = None
        self.cycle_counter = 0
        self.task_start_cycle = self.stats['cycles']
        
        # Reset state
        self.fiber_buffers = [[] for _ in range(self.radix)]
        self.fiber_buffer_heads = [0] * self.radix
        self.scaling_factors = [0.0] * self.radix
        self.fiber_ids = [None] * self.radix
        self.fiber_done = [True] * self.radix
        self.fiber_fetched = [False] * self.radix
        
        self.merger_state = {'head_indexes': [0] * self.radix, 'current_output_coord': None}
        self.accumulator_value = 0.0
        self.accumulator_coord = None
        self.output_fiber = []
        self.output_fiber_id = task.get('output_id')
        
        # Pipeline is initially empty
        self.pipeline = [None] * self.pipeline_depth
        
        # Set up the input fibers based on task type
        if task['type'] in ['direct', 'leaf']:
            # Direct or leaf task - need to fetch B rows
            for i, b_row_id in enumerate(task['B_row_ids']):
                if i < self.radix:
                    self.fiber_ids[i] = f"B_{b_row_id}"
                    self.scaling_factors[i] = task['scaling_factors'][i]
                    self.fiber_done[i] = False
                    self.fiber_fetched[i] = False
                    self._fetch_fiber(i)
        else:
            # Intermediate or root task - need to fetch partial outputs
            for i, input_id in enumerate(task['input_ids']):
                if i < self.radix:
                    self.fiber_ids[i] = input_id
                    self.scaling_factors[i] = 1.0  # Scaling factor is 1.0 for partial outputs
                    self.fiber_done[i] = False
                    self.fiber_fetched[i] = False
                    self._fetch_fiber(i)
        
        return True
    
    def _fetch_fiber(self, way_index):
        """
        Fetch a fiber from FiberCache.
        
        Args:
            way_index: Index of the way/fiber to fetch
            
        Returns:
            Boolean indicating if fetch was initiated
        """
        if self.fiber_ids[way_index] is None or self.fiber_fetched[way_index]:
            return False
        
        if self.fiber_cache:
            # Request fiber from FiberCache
            self.fiber_cache.fetch_fiber(self.fiber_ids[way_index], self.pe_id)
            self.fiber_fetched[way_index] = True
            self.stats['fiber_fetches'] += 1
            return True
        else:
            # No FiberCache - assume fiber is available
            self.fiber_fetched[way_index] = True
            return True
    
    def _load_fiber_data(self, way_index):
        """
        Load fiber data from FiberCache into the buffer.
        
        Args:
            way_index: Index of the way/fiber to load
            
        Returns:
            Boolean indicating if data was loaded
        """
        if self.fiber_ids[way_index] is None or not self.fiber_fetched[way_index]:
            return False
        
        if self.fiber_cache:
            # Try to read the fiber from FiberCache
            if self.current_task['type'] in ['intermediate', 'root']:
                # For intermediate/root tasks, consume the input fibers
                data = self.fiber_cache.consume(self.fiber_ids[way_index], self.pe_id)
            else:
                # For direct/leaf tasks, just read the B rows
                data = self.fiber_cache.read(self.fiber_ids[way_index], self.pe_id)
            
            if data is not None:
                # Process the data - in a real implementation this would decode the sparse fiber
                # For simplicity, we'll create some dummy coordinate-value pairs
                # In a full implementation, we would extract this from the actual fiber data
                self.fiber_buffers[way_index] = [(i, float(i)) for i in range(10)]
                self.stats['fiber_reads'] += 1
                return True
            else:
                # Fiber not available yet - stall
                self.stalled = True
                self.stall_reason = f"Waiting for fiber {self.fiber_ids[way_index]}"
                return False
        else:
            # No FiberCache - create dummy data
            self.fiber_buffers[way_index] = [(i, float(i)) for i in range(10)]
            return True
    
    def is_busy(self):
        """Check if the PE is currently processing a task."""
        return self.busy
    
    def is_stalled(self):
        """Check if the PE is currently stalled."""
        return self.stalled
    
    def get_stall_reason(self):
        """Get the reason for the current stall."""
        return self.stall_reason
    
    def _get_min_coord(self):
        """
        Find the minimum coordinate among all active fiber heads.
        
        Returns:
            (min_coord, ways): Tuple with minimum coordinate and list of ways that have it
        """
        min_coord = float('inf')
        ways = []
        
        for way in range(self.radix):
            # Skip if fiber is done or buffer is empty
            if (self.fiber_done[way] or 
                self.fiber_buffer_heads[way] >= len(self.fiber_buffers[way])):
                continue
                
            # Get the coordinate at head of this fiber
            head_idx = self.fiber_buffer_heads[way]
            if head_idx < len(self.fiber_buffers[way]):
                coord, _ = self.fiber_buffers[way][head_idx]
                
                if coord < min_coord:
                    min_coord = coord
                    ways = [way]
                elif coord == min_coord:
                    ways.append(way)
        
        if not ways:  # All fibers are done
            return None, []
            
        return min_coord, ways
    
    def _merge_step(self):
        """
        Perform one step of the high-radix merger.
        
        Returns:
            Tuple of (coordinate, way_index) or None if no more elements
        """
        # Check if all fibers are loaded
        all_fibers_loaded = True
        for way in range(self.radix):
            if not self.fiber_done[way] and not self.fiber_buffers[way]:
                if not self._load_fiber_data(way):
                    all_fibers_loaded = False
        
        if not all_fibers_loaded:
            # Stalled waiting for fiber data
            return None
        
        # All fibers are loaded, proceed with merge
        min_coord, ways = self._get_min_coord()
        if min_coord is None:
            return None
            
        # Select the first way that has this minimum coordinate
        selected_way = ways[0]
        
        # Increment the head index for the selected way
        self.fiber_buffer_heads[selected_way] += 1
        
        # Check if this fiber is now done
        if (self.fiber_buffer_heads[selected_way] >= 
            len(self.fiber_buffers[selected_way])):
            
            # This fiber is done
            self.fiber_done[selected_way] = True
        
        self.stats['merges'] += 1
        return (min_coord, selected_way)
    
    def _scale_step(self, coord_way):
        """
        Perform scaling operation (multiplication) on the merged element.
        
        Args:
            coord_way: Tuple of (coordinate, way_index)
            
        Returns:
            Tuple of (coordinate, scaled_value)
        """
        if coord_way is None:
            return None
            
        coord, way = coord_way
        _, value = self.fiber_buffers[way][self.fiber_buffer_heads[way] - 1]
        scaled_value = value * self.scaling_factors[way]
        
        self.stats['multiplications'] += 1
        return (coord, scaled_value)
    
    def _accumulate_step(self, coord_value):
        """
        Accumulate values with the same coordinate.
        
        Args:
            coord_value: Tuple of (coordinate, value)
            
        Returns:
            Tuple of (coordinate, accumulated_value) if a value is emitted, None otherwise
        """
        if coord_value is None:
            # If we've reached the end of all inputs and have a pending accumulation
            if self.accumulator_coord is not None:
                result = (self.accumulator_coord, self.accumulator_value)
                self.accumulator_coord = None
                self.accumulator_value = 0.0
                return result
            return None
        
        coord, value = coord_value
        
        # If this is a new coordinate, emit the previous accumulation
        if self.accumulator_coord is not None and coord != self.accumulator_coord:
            result = (self.accumulator_coord, self.accumulator_value)
            self.accumulator_coord = coord
            self.accumulator_value = value
            self.stats['additions'] += 1
            return result
        
        # First value for this coordinate or continuing accumulation
        if self.accumulator_coord is None:
            self.accumulator_coord = coord
            self.accumulator_value = value
        else:
            self.accumulator_value += value
            self.stats['additions'] += 1
        
        return None
    
    def _write_output(self):
        """
        Write the output fiber to the FiberCache.
        
        Returns:
            Boolean indicating success
        """
        if not self.output_fiber or self.output_fiber_id is None:
            return True
        
        # Convert the output fiber to a serialized format
        # In a real implementation, this would encode the sparse fiber
        output_data = bytes(f"Output fiber with {len(self.output_fiber)} elements", 'utf-8')
        
        if self.fiber_cache:
            # Write the output fiber to FiberCache
            success = self.fiber_cache.write_fiber(self.output_fiber_id, output_data, self.pe_id)
            if not success:
                self.stalled = True
                self.stall_reason = "FiberCache write failed"
                return False
        
        # Output written successfully
        return True
    
    def tick(self):
        """
        Advance the PE by one cycle.
        
        Returns:
            Status dictionary with current PE state
        """
        if not self.busy:
            self.stats['cycles'] += 1
            return {'status': 'idle', 'pe_id': self.pe_id}
        
        self.stats['cycles'] += 1
        self.cycle_counter += 1
        
        if self.stalled:
            self.stats['stall_cycles'] += 1
            # Try to resolve stall
            if self.stall_reason and "Waiting for fiber" in self.stall_reason:
                # Try loading fibers again
                all_fibers_loaded = True
                for way in range(self.radix):
                    if not self.fiber_done[way] and not self.fiber_buffers[way]:
                        if not self._load_fiber_data(way):
                            all_fibers_loaded = False
                
                if all_fibers_loaded:
                    self.stalled = False
                    self.stall_reason = None
            elif self.stall_reason == "FiberCache write failed":
                # Try writing output again
                if self._write_output():
                    self.stalled = False
                    self.stall_reason = None
            
            return {
                'status': 'stalled',
                'pe_id': self.pe_id,
                'reason': self.stall_reason,
                'cycle': self.cycle_counter
            }
        
        self.stats['busy_cycles'] += 1
        
        # Process pipeline stages in reverse order to avoid data hazards
        # Stage 3: Accumulate
        if self.pipeline[3] is not None:
            output = self._accumulate_step(self.pipeline[3])
            if output is not None:
                self.output_fiber.append(output)
                self.stats['elements_processed'] += 1
        
        # Stage 2: Scale
        if self.pipeline[2] is not None:
            self.pipeline[3] = self._scale_step(self.pipeline[2])
        else:
            self.pipeline[3] = None
        
        # Stage 1: Merge
        if self.pipeline[1] is not None:
            self.pipeline[2] = self.pipeline[1]
        else:
            self.pipeline[2] = None
        
        # Stage 0: Fetch new element
        self.pipeline[1] = self._merge_step()
        
        # Check if we're done
        all_done = all(self.fiber_done) or all(
            self.fiber_buffer_heads[i] >= len(self.fiber_buffers[i])
            for i in range(self.radix) if not self.fiber_done[i]
        )
        
        pipeline_empty = all(stage is None for stage in self.pipeline[1:])
        
        if all_done and pipeline_empty:
            # Flush any remaining value in accumulator
            if self.accumulator_coord is not None:
                self.output_fiber.append((self.accumulator_coord, self.accumulator_value))
                self.accumulator_coord = None
                self.accumulator_value = 0.0
            
            # Write output fiber to FiberCache
            if self._write_output():
                # Task completed
                self.busy = False
                self.stats['tasks_completed'] += 1
        
        return {
            'status': 'busy' if self.busy else 'completed',
            'pe_id': self.pe_id,
            'cycle': self.cycle_counter,
            'pipeline_state': [
                'empty' if stage is None else f"processing coord {stage[0] if isinstance(stage, tuple) else 'unknown'}" 
                for stage in self.pipeline
            ],
            'output_size': len(self.output_fiber),
            'task_id': self.task_id,
            'task_latency': self.stats['cycles'] - self.task_start_cycle if not self.busy else None
        }
    
    def get_output_fiber(self):
        """
        Get the output fiber produced by this PE.
        
        Returns:
            Output fiber (list of coordinate-value tuples)
        """
        return self.output_fiber
    
    def get_task_id(self):
        """
        Get the ID of the currently executing task.
        
        Returns:
            Task ID or None if idle
        """
        return self.task_id if self.busy else None
    
    def get_stats(self):
        """
        Get performance statistics for this PE.
        
        Returns:
            Dictionary of statistics
        """
        return self.stats


class ProcessingElementArray:
    """
    Manages an array of Processing Elements for the Gamma accelerator.
    """
    def __init__(self, num_pes=32, radix=64, fiber_cache=None):
        """
        Initialize an array of Processing Elements.
        
        Args:
            num_pes: Number of PEs in the array
            radix: Merger radix for each PE
            fiber_cache: Reference to the FiberCache
        """
        self.pes = [ProcessingElement(pe_id=i, radix=radix, fiber_cache=fiber_cache) for i in range(num_pes)]
        self.num_pes = num_pes
        self.fiber_cache = fiber_cache
        self.stats = {
            'total_cycles': 0,
            'pe_utilization': 0.0,
            'elements_processed': 0,
            'idle_pe_cycles': 0,
            'stalled_pe_cycles': 0,
            'tasks_completed': 0
        }
    
    def assign_task(self, pe_id, task):
        """
        Assign a task to a specific PE.
        
        Args:
            pe_id: ID of the PE to assign the task to
            task: Task description
            
        Returns:
            Boolean indicating if the task was assigned
        """
        if 0 <= pe_id < self.num_pes:
            return self.pes[pe_id].set_task(task)
        return False
    
    def get_idle_pe(self):
        """
        Find the ID of an idle PE.
        
        Returns:
            PE ID or None if all PEs are busy
        """
        for i, pe in enumerate(self.pes):
            if not pe.is_busy():
                return i
        return None
    
    def get_completed_tasks(self):
        """
        Get IDs of tasks that have completed this cycle.
        
        Returns:
            List of (PE ID, task ID) tuples for completed tasks
        """
        completed = []
        for i, pe in enumerate(self.pes):
            status = pe.tick()
            if status['status'] == 'completed':
                completed.append((i, status['task_id']))
                self.stats['tasks_completed'] += 1
        
        return completed
    
    def tick(self):
        """
        Advance all PEs by one cycle.
        
        Returns:
            List of status dictionaries from all PEs
        """
        self.stats['total_cycles'] += 1
        busy_count = 0
        stalled_count = 0
        
        results = []
        for pe in self.pes:
            status = pe.tick()
            results.append(status)
            
            if status['status'] == 'busy':
                busy_count += 1
            elif status['status'] == 'stalled':
                stalled_count += 1
                self.stats['stalled_pe_cycles'] += 1
            else:
                self.stats['idle_pe_cycles'] += 1
        
        # Update utilization statistics
        self.stats['pe_utilization'] = busy_count / self.num_pes
        
        return results
    
    def get_pe_status(self, pe_id):
        """
        Get the status of a specific PE.
        
        Args:
            pe_id: ID of the PE
            
        Returns:
            Status dictionary
        """
        if 0 <= pe_id < self.num_pes:
            pe = self.pes[pe_id]
            return {
                'busy': pe.is_busy(),
                'stalled': pe.is_stalled(),
                'stall_reason': pe.get_stall_reason(),
                'task_id': pe.get_task_id(),
                'stats': pe.get_stats()
            }
        return None
    
    def get_aggregate_stats(self):
        """
        Get aggregated statistics across all PEs.
        
        Returns:
            Dictionary of aggregate statistics
        """
        total_elements = 0
        total_multiplications = 0
        total_additions = 0
        total_merges = 0
        total_fiber_reads = 0
        max_cycles = 0
        
        for pe in self.pes:
            pe_stats = pe.get_stats()
            total_elements += pe_stats['elements_processed']
            total_multiplications += pe_stats['multiplications']
            total_additions += pe_stats['additions']
            total_merges += pe_stats['merges']
            total_fiber_reads += pe_stats['fiber_reads']
            max_cycles = max(max_cycles, pe_stats['cycles'])
        
        self.stats['elements_processed'] = total_elements
        self.stats['tasks_completed'] = sum(pe.get_stats()['tasks_completed'] for pe in self.pes)
        
        # Calculate PE utilization over time
        if self.stats['total_cycles'] > 0:
            overall_utilization = (self.stats['total_cycles'] * self.num_pes - 
                                  self.stats['idle_pe_cycles'] - 
                                  self.stats['stalled_pe_cycles']) / (self.stats['total_cycles'] * self.num_pes)
        else:
            overall_utilization = 0
            
        
        return {
            'total_cycles': self.stats['total_cycles'],
            'max_pe_cycles': max_cycles,
            'overall_pe_utilization': overall_utilization,
            'current_pe_utilization': self.stats['pe_utilization'],
            'total_elements_processed': total_elements,
            'total_multiplications': total_multiplications,
            'total_additions': total_additions,
            'total_merges': total_merges,
            'total_fiber_reads': total_fiber_reads,
            'idle_pe_cycles': self.stats['idle_pe_cycles'],
            'stalled_pe_cycles': self.stats['stalled_pe_cycles'],
            'tasks_completed': self.stats['tasks_completed']
        }