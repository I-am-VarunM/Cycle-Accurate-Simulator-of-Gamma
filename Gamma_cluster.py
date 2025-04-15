"""
Cycle-Accurate Simulator for Gamma Architecture

This simulator models the full Gamma architecture from the paper:
"Gamma: Leveraging Gustavson's Algorithm to Accelerate Sparse Matrix Multiplication"

It includes 32 Processing Elements (PEs), FiberCache, and scheduler.
"""

from Gamma_cycleaccurate import Fiber, FiberBuffer, GammaPE

import heapq
import random
import numpy as np
import time
from collections import defaultdict, deque

class FiberCache:
    """
    Models the FiberCache structure from the Gamma architecture.
    Organized as a cache but managed explicitly to buffer fibers.
    """
    def __init__(self, size_mb=3, line_size_bytes=64, ways=16, banks=48):
        self.size_bytes = size_mb * 1024 * 1024  # Convert MB to bytes
        self.line_size_bytes = line_size_bytes
        self.ways = ways
        self.banks = banks
        
        # Calculate lines per set
        total_lines = self.size_bytes // self.line_size_bytes
        self.sets = total_lines // self.ways
        
        # Initialize cache structure: [sets][ways]
        self.tags = [[-1 for _ in range(self.ways)] for _ in range(self.sets)]
        self.data = [[None for _ in range(self.ways)] for _ in range(self.sets)]
        self.priorities = [[0 for _ in range(self.ways)] for _ in range(self.sets)]
        self.valid = [[False for _ in range(self.ways)] for _ in range(self.sets)]
        self.dirty = [[False for _ in range(self.ways)] for _ in range(self.sets)]
        
        # SRRIP replacement state (2-bit counter)
        self.srrip = [[3 for _ in range(self.ways)] for _ in range(self.sets)]
        
        # Statistics
        self.stats = {
            "reads": 0,
            "writes": 0,
            "fetches": 0,
            "consumes": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "B_fibers_stored": 0,
            "partial_results_stored": 0,
            "capacity_utilization": 0.0,  # Percentage of cache used
            "bank_conflicts": 0,
            "read_bandwidth_used": 0,
            "write_bandwidth_used": 0,
            "total_bandwidth_used": 0,
            "cycles_active": 0
        }
        
        # Per-cycle access tracking
        self.current_cycle = 0
        self.cycle_bank_accesses = {}  # maps cycle to set of banks accessed
        
        # Fiber storage - maps fiber_id to data
        self.fibers = {}
        self.fiber_addr_map = {}  # Maps fiber_id to address range
        self.addr_fiber_map = {}  # Maps address to fiber_id
        
        # Track partial result fibers
        self.partial_result_fibers = set()
        
        # Pending fetches - simulate memory latency
        self.pending_fetches = {}  # Map of fetch_id to (cycle, address)
        self.next_fetch_id = 0
        self.memory_latency = 80  # 80 cycles latency for memory access
        
    def _get_set_index(self, addr):
        """Get the set index from an address"""
        return (addr // self.line_size_bytes) % self.sets
        
    def _get_tag(self, addr):
        """Get the tag from an address"""
        return addr // (self.line_size_bytes * self.sets)
        
    def _get_bank(self, addr):
        """Get the bank index from an address"""
        return addr % self.banks
        
    def _find_line(self, addr):
        """Find a line in the cache given an address"""
        set_idx = self._get_set_index(addr)
        tag = self._get_tag(addr)
        
        for way in range(self.ways):
            if self.valid[set_idx][way] and self.tags[set_idx][way] == tag:
                return set_idx, way
                
        return None, None
        
    def _find_victim(self, set_idx):
        """Find a victim line for replacement using SRRIP policy"""
        # First, look for invalid lines
        for way in range(self.ways):
            if not self.valid[set_idx][way]:
                return way
                
        # Look for lowest priority first
        min_priority = float('inf')
        min_ways = []
        
        for way in range(self.ways):
            if self.priorities[set_idx][way] < min_priority:
                min_priority = self.priorities[set_idx][way]
                min_ways = [way]
            elif self.priorities[set_idx][way] == min_priority:
                min_ways.append(way)
                
        if len(min_ways) == 1:
            return min_ways[0]
            
        # Break ties with SRRIP
        max_srrip = -1
        victim_way = -1
        
        for way in min_ways:
            if self.srrip[set_idx][way] > max_srrip:
                max_srrip = self.srrip[set_idx][way]
                victim_way = way
                
        return victim_way
        
    def _evict_line(self, set_idx, way):
        """Evict a line from the cache"""
        if self.valid[set_idx][way] and self.dirty[set_idx][way]:
            # Would write back to memory in a real system
            # Track bandwidth used for writeback
            self.stats["write_bandwidth_used"] += self.line_size_bytes
            self.stats["total_bandwidth_used"] += self.line_size_bytes
            
        self.valid[set_idx][way] = False
        self.dirty[set_idx][way] = False
        self.stats["evictions"] += 1
        
    def _update_srrip(self, set_idx, way):
        """Update SRRIP state on access"""
        self.srrip[set_idx][way] = 0  # Set to near-immediate re-reference
    
    def _check_bank_conflict(self, addr):
        """
        Check if accessing this address causes a bank conflict in the current cycle
        Returns True if there's a conflict, False otherwise
        """
        bank = self._get_bank(addr)
        
        # Initialize bank access tracking for this cycle if needed
        if self.current_cycle not in self.cycle_bank_accesses:
            self.cycle_bank_accesses[self.current_cycle] = set()
            
        # Check if this bank was already accessed this cycle
        if bank in self.cycle_bank_accesses[self.current_cycle]:
            self.stats["bank_conflicts"] += 1
            return True
            
        # Record bank access
        self.cycle_bank_accesses[self.current_cycle].add(bank)
        return False
        
    def fetch(self, fiber_id, address, size, is_partial_result=False):
        """
        Fetch data into cache ahead of time (decoupled access)
        Returns a fetch ID that can be used to check if fetch is complete
        """
        self.stats["fetches"] += 1
        fetch_id = self.next_fetch_id
        self.next_fetch_id += 1
        
        # Store fetch information
        self.pending_fetches[fetch_id] = (self.memory_latency, address, size)
        
        # Map fiber to address range for future reference
        self.fiber_addr_map[fiber_id] = (address, address + size)
        
        # If this is a partial result fiber, track it
        if is_partial_result:
            self.partial_result_fibers.add(fiber_id)
        
        # Map addresses to fiber for lookups
        elem_size = 16  # Assuming 16 bytes per element (coordinate + value)
        for i in range(size // elem_size):
            elem_addr = address + i * elem_size
            self.addr_fiber_map[elem_addr] = fiber_id
            
        return fetch_id
        
    def check_fetch_complete(self, fetch_id, current_cycle):
        """Check if a fetch is complete based on cycles passed"""
        if fetch_id not in self.pending_fetches:
            return True
            
        latency_remaining, address, size = self.pending_fetches[fetch_id]
        latency_remaining -= 1
        
        if latency_remaining <= 0:
            # Fetch is complete, bring data into cache
            # Track read bandwidth used for fetch
            self.stats["read_bandwidth_used"] += size
            self.stats["total_bandwidth_used"] += size
            
            # For each cache line in the fetched region
            for line_offset in range(0, size, self.line_size_bytes):
                line_addr = address + line_offset
                
                # Find if line is already in cache or needs replacement
                set_idx = self._get_set_index(line_addr)
                tag = self._get_tag(line_addr)
                
                existing_set, existing_way = self._find_line(line_addr)
                
                if existing_set is not None:
                    # Already in cache, update priority
                    self.priorities[existing_set][existing_way] += 1
                    self._update_srrip(existing_set, existing_way)
                else:
                    # Need replacement
                    way = self._find_victim(set_idx)
                    self._evict_line(set_idx, way)
                    
                    # Install new line
                    self.tags[set_idx][way] = tag
                    self.valid[set_idx][way] = True
                    self.dirty[set_idx][way] = False
                    self.priorities[set_idx][way] = 1  # Start with priority 1
                    self._update_srrip(set_idx, way)
            
            # Fetch is now complete
            del self.pending_fetches[fetch_id]
            return True
        else:
            # Update remaining latency
            self.pending_fetches[fetch_id] = (latency_remaining, address, size)
            return False
            
    def read(self, addr):
        """Read data from the cache"""
        self.stats["reads"] += 1
        
        # Check for bank conflicts
        self._check_bank_conflict(addr)
        
        set_idx, way = self._find_line(addr)
        
        if set_idx is not None:
            # Cache hit
            self.stats["hits"] += 1
            self.priorities[set_idx][way] -= 1  # Decrement priority on read
            self._update_srrip(set_idx, way)
            
            # Return the data from cache
            return self.data[set_idx][way]
        else:
            # Cache miss
            self.stats["misses"] += 1
            
            # In a real system, this would trigger a fetch
            # For simulation, we'll track it but return None
            # Caller should handle miss appropriately
            
            # Track read bandwidth used for line fetch
            self.stats["read_bandwidth_used"] += self.line_size_bytes
            self.stats["total_bandwidth_used"] += self.line_size_bytes
            
            return None
            
    def write(self, addr, data, is_partial_result=False):
        """Write data to the cache without fetching from memory"""
        self.stats["writes"] += 1
        
        # Check for bank conflicts
        self._check_bank_conflict(addr)
        
        set_idx = self._get_set_index(addr)
        tag = self._get_tag(addr)
        
        # Look for existing entry
        existing_set, existing_way = self._find_line(addr)
        
        if existing_set is not None:
            # Update existing entry - cache hit
            self.stats["hits"] += 1
            self.data[existing_set][existing_way] = data
            self.dirty[existing_set][existing_way] = True
            self._update_srrip(existing_set, existing_way)
        else:
            # Cache miss - allocate new entry
            self.stats["misses"] += 1
            way = self._find_victim(set_idx)
            self._evict_line(set_idx, way)
            
            # Install new line
            self.tags[set_idx][way] = tag
            self.data[set_idx][way] = data
            self.valid[set_idx][way] = True
            self.dirty[set_idx][way] = True
            self._update_srrip(set_idx, way)
            
            # Update statistics
            if is_partial_result:
                self.stats["partial_results_stored"] += 1
            else:
                self.stats["B_fibers_stored"] += 1
            
    def consume(self, addr):
        """Read and invalidate line (for partial results)"""
        self.stats["consumes"] += 1
        
        # Check for bank conflicts
        self._check_bank_conflict(addr)
        
        set_idx, way = self._find_line(addr)
        
        if set_idx is not None:
            # Cache hit
            self.stats["hits"] += 1
            data = self.data[set_idx][way]
            
            # Invalidate the line
            self.valid[set_idx][way] = False
            
            return data
        else:
            # Cache miss
            self.stats["misses"] += 1
            return None
    
    def get_fiber(self, fiber_id):
        """
        Get a fiber by ID if it's in the cache
        Returns a list of (coord, value) tuples if found, None otherwise
        """
        if fiber_id not in self.fiber_addr_map:
            return None
            
        start_addr, end_addr = self.fiber_addr_map[fiber_id]
        
        # Check if all elements of the fiber are in cache
        elem_size = 16  # Assuming 16 bytes per element
        elements = []
        all_in_cache = True
        
        for addr in range(start_addr, end_addr, elem_size):
            set_idx, way = self._find_line(addr)
            if set_idx is None:
                all_in_cache = False
                break
            else:
                elements.append(self.data[set_idx][way])
        
        if all_in_cache:
            return elements
        else:
            return None
            
    def update_cycles(self, current_cycle):
        """Update pending fetches based on cycles and track active cycles"""
        self.current_cycle = current_cycle
        self.stats["cycles_active"] += 1
        
        # Update capacity utilization statistic
        total_valid_lines = 0
        for set_idx in range(self.sets):
            for way in range(self.ways):
                if self.valid[set_idx][way]:
                    total_valid_lines += 1
        
        self.stats["capacity_utilization"] = total_valid_lines / (self.sets * self.ways) * 100
        
        # Process pending fetches
        completed_fetches = []
        
        for fetch_id in list(self.pending_fetches.keys()):
            if self.check_fetch_complete(fetch_id, current_cycle):
                completed_fetches.append(fetch_id)
                
        # Clean up old bank access tracking (keep last 10 cycles)
        for cycle in list(self.cycle_bank_accesses.keys()):
            if cycle < current_cycle - 10:
                del self.cycle_bank_accesses[cycle]
                
        return completed_fetches
        
    def print_stats(self):
        """Print cache statistics"""
        print("\nFiberCache Statistics:")
        print(f"Size: {self.size_bytes / (1024 * 1024):.2f} MB, {self.sets} sets, {self.ways} ways, {self.banks} banks")
        print(f"Capacity utilization: {self.stats['capacity_utilization']:.2f}%")
        print(f"Active cycles: {self.stats['cycles_active']}")
        print(f"B fibers stored: {self.stats['B_fibers_stored']}")
        print(f"Partial results stored: {self.stats['partial_results_stored']}")
        print(f"Reads: {self.stats['reads']}")
        print(f"Writes: {self.stats['writes']}")
        print(f"Fetches: {self.stats['fetches']}")
        print(f"Consumes: {self.stats['consumes']}")
        print(f"Hits: {self.stats['hits']}")
        print(f"Misses: {self.stats['misses']}")
        if self.stats['reads'] + self.stats['writes'] > 0:
            hit_rate = self.stats['hits'] / (self.stats['reads'] + self.stats['writes']) * 100
            print(f"Hit rate: {hit_rate:.2f}%")
        print(f"Evictions: {self.stats['evictions']}")
        print(f"Bank conflicts: {self.stats['bank_conflicts']}")
        print(f"Memory bandwidth used: {self.stats['total_bandwidth_used'] / (1024 * 1024):.2f} MB")
        print(f"  - Read: {self.stats['read_bandwidth_used'] / (1024 * 1024):.2f} MB")
        print(f"  - Write: {self.stats['write_bandwidth_used'] / (1024 * 1024):.2f} MB")


class Scheduler:
    """
    Enhanced Scheduler for the Gamma architecture.
    Assigns tasks to PEs and manages execution flow with proper fiber tracking.
    """
    def __init__(self, num_pes=32, pe_radix=64):
        self.num_pes = num_pes
        self.pe_radix = pe_radix
        self.pe_status = [False] * num_pes  # False = idle, True = busy
        self.task_queue = deque()  # Tasks waiting to be assigned
        
        # Enhanced task tracking
        self.active_tasks = {}  # Maps task_id to task
        self.completed_tasks = {}  # Maps task_id to completed task
        self.task_dependencies = {}  # Maps task_id to list of dependency task_ids
        self.pe_task_map = {}  # Maps PE index to current task_id
        
        # Enhanced fiber tracking
        self.fiber_locations = {}  # Maps fiber_id to PE index or "memory"
        self.fiber_metadata = {}  # Maps fiber_id to (size, is_partial, addr, status)
        
        # For task tree creation
        self.next_task_id = 0
        
        # Stats
        self.stats = {
            "total_tasks": 0,
            "leaf_tasks": 0,
            "merge_tasks": 0,
            "multi_round_tasks": 0,
            "pe_utilization": [0] * num_pes,
            "task_execution_cycles": {},  # Maps task type to cycles
            "fiber_reuse_count": {},  # Maps fiber_id to reuse count
            "scheduler_overhead_cycles": 0
        }
        
    def assign_task_id(self):
        """Generate a unique task ID"""
        task_id = self.next_task_id
        self.next_task_id += 1
        return task_id
        
    def register_fiber(self, fiber_id, size, addr, is_partial=False, owner_pe=None):
        """Register a fiber with the scheduler"""
        status = "memory" if owner_pe is None else f"pe_{owner_pe}"
        self.fiber_metadata[fiber_id] = (size, is_partial, addr, status)
        self.fiber_locations[fiber_id] = owner_pe if owner_pe is not None else "memory"
        
        # Initialize reuse counter
        if fiber_id not in self.stats["fiber_reuse_count"]:
            self.stats["fiber_reuse_count"][fiber_id] = 0
            
    def create_task_tree(self, A_row, B_fibers, system=None):
        """
        Create a balanced, top-full tree of tasks for processing an A row
        
        Args:
            A_row: The row of matrix A to process
            B_fibers: The fibers of matrix B needed for this row
            system: Optional reference to the GammaSystem for ID/address assignment
            
        Returns:
            The root of the task tree
        """
        self.stats["total_tasks"] += 1
        
        # If fibers need IDs or addresses, assign them if system is provided
        if system:
            for fiber in B_fibers:
                if not hasattr(fiber, 'id'):
                    fiber.id = system.assign_fiber_id()
                if not hasattr(fiber, 'addr'):
                    fiber.addr = system.assign_fiber_addr(len(fiber.coords))
                    
                # Register fiber with scheduler
                self.register_fiber(fiber.id, len(fiber.coords), fiber.addr)
        
        if len(B_fibers) <= self.pe_radix:
            # Can be processed in a single pass
            task_id = self.assign_task_id()
            self.stats["leaf_tasks"] += 1
            
            task = {
                "task_id": task_id,
                "type": "leaf",
                "is_root": True,
                "A_row": A_row,
                "B_fibers": B_fibers,
                "output_row_index": A_row.index if hasattr(A_row, 'index') else None,
                "children": [],
                "dependencies": [],
                "status": "pending"
            }
            
            self.active_tasks[task_id] = task
            return task
            
        # Need multiple passes - count as multi-round task
        self.stats["multi_round_tasks"] += 1
        
        # Group B_fibers into chunks of pe_radix
        chunks = []
        for i in range(0, len(B_fibers), self.pe_radix):
            chunks.append(B_fibers[i:i+self.pe_radix])
            
        # Create leaf tasks for each chunk
        leaf_tasks = []
        for chunk in chunks:
            task_id = self.assign_task_id()
            self.stats["leaf_tasks"] += 1
            
            leaf_task = {
                "task_id": task_id,
                "type": "leaf",
                "is_root": False,
                "A_row": A_row,
                "B_fibers": chunk,
                "children": [],
                "dependencies": [],
                "status": "pending"
            }
            
            self.active_tasks[task_id] = leaf_task
            leaf_tasks.append(leaf_task)
            
        # Create a balanced tree to merge partial results
        root_task = self._build_balanced_tree(leaf_tasks, A_row)
        return root_task
        
    def _build_balanced_tree(self, tasks, A_row):
        """Recursively build a balanced tree of merge tasks"""
        if len(tasks) == 1:
            # If this is the only task and it's a leaf task, mark it as root
            if tasks[0]["type"] == "leaf":
                tasks[0]["is_root"] = True
                tasks[0]["output_row_index"] = A_row.index if hasattr(A_row, 'index') else None
            return tasks[0]
            
        # Group tasks into chunks of pe_radix for merging
        chunks = []
        for i in range(0, len(tasks), self.pe_radix):
            chunks.append(tasks[i:i+self.pe_radix])
            
        # Create merge tasks for each chunk
        merge_tasks = []
        for chunk in chunks:
            if len(chunk) == 1:
                merge_tasks.append(chunk[0])
                continue
                
            task_id = self.assign_task_id()
            self.stats["merge_tasks"] += 1
            
            merge_task = {
                "task_id": task_id,
                "type": "merge",
                "is_root": False,
                "children": chunk,
                "dependencies": [child["task_id"] for child in chunk],
                "status": "pending"
            }
            
            # Register dependencies
            for child in chunk:
                if child["task_id"] not in self.task_dependencies:
                    self.task_dependencies[child["task_id"]] = []
                self.task_dependencies[child["task_id"]].append(task_id)
            
            self.active_tasks[task_id] = merge_task
            merge_tasks.append(merge_task)
            
        # If we have only one merge task, we're done
        if len(merge_tasks) == 1:
            # Mark as root task
            merge_tasks[0]["is_root"] = True
            merge_tasks[0]["output_row_index"] = A_row.index if hasattr(A_row, 'index') else None
            return merge_tasks[0]
            
        # Otherwise, recurse to create higher level merge tasks
        return self._build_balanced_tree(merge_tasks, A_row)
        
    def schedule_row(self, A_row, B_rows, system=None):
        """Schedule processing of an A row with corresponding B rows"""
        task_tree = self.create_task_tree(A_row, B_rows, system)
        self._enqueue_ready_tasks(task_tree)
        
    def _enqueue_ready_tasks(self, task):
        """Enqueue tasks with no dependencies"""
        if task["status"] != "pending":
            return
            
        if not task.get("dependencies", []):
            # No dependencies, can be enqueued
            self.task_queue.append(task)
            task["status"] = "queued"
        else:
            # Has dependencies, will be enqueued when dependencies complete
            pass
        
        # Recursively enqueue children
        for child in task.get("children", []):
            self._enqueue_ready_tasks(child)
            
    def assign_tasks(self, pes, system=None):
        """
        Assign tasks to idle PEs
        
        Args:
            pes: List of PE objects
            system: Optional reference to GammaSystem for fiber registration
            
        Returns:
            Number of tasks assigned
        """
        assigned_count = 0
        
        for pe_idx in range(self.num_pes):
            # Skip if PE is busy or invalid
            if self.pe_status[pe_idx] or pes[pe_idx] is None:
                continue
                
            # Check if PE is idle
            if (pes[pe_idx].fetch_stage is None and 
                pes[pe_idx].merge_stage is None and 
                pes[pe_idx].multiply_stage is None and 
                pes[pe_idx].accumulate_stage is None):
                
                # Try to assign a task
                if len(self.task_queue) > 0:
                    task = self.task_queue.popleft()
                    task["status"] = "running"
                    task["assigned_pe"] = pe_idx
                    task["start_cycle"] = system.cycle if system else 0
                    
                    # Record assignment
                    self.pe_task_map[pe_idx] = task["task_id"]
                    
                    # Setup based on task type
                    if task["type"] == "leaf":
                        # For leaf tasks, assign B fibers
                        if system:
                            # Create output fiber
                            output_fiber_id = system.assign_fiber_id()
                            output_fiber_addr = system.assign_fiber_addr(100)  # Estimate initial size
                            
                            # Register with scheduler
                            self.register_fiber(
                                output_fiber_id, 
                                100,  # Estimated size 
                                output_fiber_addr,
                                is_partial=not task["is_root"],
                                owner_pe=pe_idx
                            )
                            
                            # Set task output information
                            task["output_id"] = output_fiber_id
                            task["output_addr"] = output_fiber_addr
                            
                            # Setup PE with fiber cache information
                            fiber_ids = [fiber.id for fiber in task["B_fibers"]]
                            fiber_addrs = [fiber.addr for fiber in task["B_fibers"]]
                            
                            # Update fiber reuse counters
                            for fiber_id in fiber_ids:
                                self.stats["fiber_reuse_count"][fiber_id] += 1
                            
                            # Setup PE
                            pes[pe_idx].set_input_fibers(
                                fibers=task["B_fibers"],
                                fiber_cache=system.fibercache,
                                fiber_ids=fiber_ids,
                                fiber_addrs=fiber_addrs
                            )
                            pes[pe_idx].output_fiber_id = output_fiber_id
                            pes[pe_idx].output_fiber_addr = output_fiber_addr
                        else:
                            # Simple setup without system
                            pes[pe_idx].set_input_fibers(task["B_fibers"])
                            
                    elif task["type"] == "merge":
                        # For merge tasks, gather partial results
                        input_fibers = []
                        fiber_ids = []
                        fiber_addrs = []
                        
                        for child in task["children"]:
                            if "output_id" in child and "output_addr" in child:
                                # Create a fiber placeholder
                                fiber = Fiber([], [], scaling_factor=1.0)
                                fiber.id = child["output_id"]
                                fiber.addr = child["output_addr"]
                                input_fibers.append(fiber)
                                fiber_ids.append(fiber.id)
                                fiber_addrs.append(fiber.addr)
                                
                                # Update fiber reuse counter
                                self.stats["fiber_reuse_count"][fiber.id] += 1
                        
                        if system:
                            # Create output fiber
                            output_fiber_id = system.assign_fiber_id()
                            output_fiber_addr = system.assign_fiber_addr(200)  # Larger estimate for merge result
                            
                            # Register with scheduler
                            self.register_fiber(
                                output_fiber_id, 
                                200,  # Estimated size
                                output_fiber_addr,
                                is_partial=not task["is_root"],
                                owner_pe=pe_idx
                            )
                            
                            # Set task output information
                            task["output_id"] = output_fiber_id
                            task["output_addr"] = output_fiber_addr
                            
                            # Setup PE
                            pes[pe_idx].set_input_fibers(
                                fibers=input_fibers,
                                fiber_cache=system.fibercache,
                                fiber_ids=fiber_ids,
                                fiber_addrs=fiber_addrs
                            )
                            pes[pe_idx].output_fiber_id = output_fiber_id
                            pes[pe_idx].output_fiber_addr = output_fiber_addr
                        else:
                            # Simple setup without system
                            pes[pe_idx].set_input_fibers(input_fibers)
                    
                    # Mark PE as busy
                    self.pe_status[pe_idx] = True
                    assigned_count += 1
                    
        return assigned_count
                    
    def update_pe_status(self, pes):
        """
        Update status of PEs and collect completed tasks
        
        Args:
            pes: List of PE objects
            
        Returns:
            List of newly completed task IDs
        """
        completed_tasks = []
        
        for pe_idx in range(self.num_pes):
            if self.pe_status[pe_idx] and pes[pe_idx] is not None:
                # Check if PE is now idle
                if (pes[pe_idx].fetch_stage is None and 
                    pes[pe_idx].merge_stage is None and 
                    pes[pe_idx].multiply_stage is None and 
                    pes[pe_idx].accumulate_stage is None):
                    
                    # PE has completed its task
                    self.pe_status[pe_idx] = False
                    self.stats["pe_utilization"][pe_idx] += 1
                    
                    # Get completed task
                    if pe_idx in self.pe_task_map:
                        task_id = self.pe_task_map[pe_idx]
                        task = self.active_tasks.get(task_id)
                        
                        if task:
                            # Get result fiber
                            result_fiber = pes[pe_idx].get_result_fiber()
                            
                            # Update task information
                            task["status"] = "completed"
                            task["result_size"] = result_fiber.size
                            task["end_cycle"] = pes[pe_idx].cycle
                            
                            # Update statistics
                            task_type = task["type"]
                            if task_type not in self.stats["task_execution_cycles"]:
                                self.stats["task_execution_cycles"][task_type] = []
                            self.stats["task_execution_cycles"][task_type].append(task["end_cycle"] - task["start_cycle"])
                            
                            # Move to completed tasks
                            self.completed_tasks[task_id] = task
                            del self.active_tasks[task_id]
                            
                            # Remove from PE task map
                            del self.pe_task_map[pe_idx]
                            
                            # Add to completed list
                            completed_tasks.append(task_id)
                            
                            # Check if any dependent tasks can now be scheduled
                            self._check_dependent_tasks(task_id)
        
        return completed_tasks
    
    def _check_dependent_tasks(self, completed_task_id):
        """Check if any tasks dependent on the completed task can now be scheduled"""
        if completed_task_id in self.task_dependencies:
            for dependent_task_id in self.task_dependencies[completed_task_id]:
                dependent_task = self.active_tasks.get(dependent_task_id)
                
                if dependent_task and dependent_task["status"] == "pending":
                    # Check if all dependencies are completed
                    all_dependencies_met = True
                    for dep_id in dependent_task["dependencies"]:
                        if dep_id not in self.completed_tasks:
                            all_dependencies_met = False
                            break
                            
                    if all_dependencies_met:
                        # All dependencies are met, can enqueue this task
                        self.task_queue.append(dependent_task)
                        dependent_task["status"] = "queued"
                
    def is_schedule_complete(self):
        """Check if all tasks have been processed"""
        return len(self.task_queue) == 0 and not any(self.pe_status) and len(self.active_tasks) == 0
        
    def print_stats(self):
        """Print scheduler statistics"""
        print("\nScheduler Statistics:")
        print(f"Total tasks: {self.stats['total_tasks']}")
        print(f"  - Leaf tasks: {self.stats['leaf_tasks']}")
        print(f"  - Merge tasks: {self.stats['merge_tasks']}")
        print(f"Multi-round tasks: {self.stats['multi_round_tasks']}")
        
        # Calculate average task execution time by type
        print("\nAverage task execution cycles:")
        for task_type, cycles_list in self.stats["task_execution_cycles"].items():
            if cycles_list:
                avg_cycles = sum(cycles_list) / len(cycles_list)
                print(f"  - {task_type}: {avg_cycles:.2f} cycles")
        
        # Calculate PE utilization
        total_util = sum(self.stats["pe_utilization"])
        avg_util = total_util / max(1, self.num_pes)
        print(f"\nAverage PE utilization: {avg_util:.2f}")
        
        # Print fiber reuse statistics
        total_fibers = len(self.stats["fiber_reuse_count"])
        if total_fibers > 0:
            total_reuses = sum(self.stats["fiber_reuse_count"].values())
            avg_reuse = total_reuses / total_fibers
            print(f"\nFiber statistics:")
            print(f"  - Total unique fibers: {total_fibers}")
            print(f"  - Average reuse per fiber: {avg_reuse:.2f}")
            
            # Find most reused fibers
            top_reused = sorted(self.stats["fiber_reuse_count"].items(), key=lambda x: x[1], reverse=True)[:5]
            if top_reused:
                print(f"  - Top reused fibers:")
                for fiber_id, reuse_count in top_reused:
                    print(f"    - Fiber {fiber_id}: {reuse_count} reuses")


class GammaSystem:
    """
    Top-level class that models the full Gamma system with 32 PEs, FiberCache, and scheduler
    Updated for proper FiberCache integration
    """
    def __init__(self, num_pes=32, pe_radix=64, cache_size_mb=3):
        self.num_pes = num_pes
        self.pe_radix = pe_radix
        
        # Initialize components
        self.pes = [GammaPE(radix=pe_radix) for _ in range(num_pes)]
        self.fibercache = FiberCache(size_mb=cache_size_mb)
        self.scheduler = Scheduler(num_pes=num_pes, pe_radix=pe_radix)
        
        # Connect FiberCache with PEs
        for pe in self.pes:
            pe.fiber_cache = self.fibercache
        
        # System stats
        self.cycle = 0
        self.memory_bw_limit = 128  # 128 GB/s
        self.memory_bw_utilized = 0
        self.system_stats = {
            "total_elements_processed": 0,
            "active_pe_cycles": 0,
            "idle_pe_cycles": 0,
            "pe_utilization": 0.0,
            "memory_bandwidth_utilization": 0.0,
            "rows_processed": 0,
            "memory_traffic_reduction": 0.0  # Compared to compulsory traffic
        }
        
        # Addresses and IDs for fibers
        self.next_fiber_id = 1
        self.next_addr = 1000000  # Start at a high address to avoid conflicts
        
        # Track fiber assignments
        self.fiber_assignments = {}  # Maps fiber id to PE index
        self.fiber_info = {}  # Maps fiber id to (size, is_partial_result)
        
    def assign_fiber_id(self):
        """Generate a unique fiber ID"""
        fiber_id = self.next_fiber_id
        self.next_fiber_id += 1
        return fiber_id
        
    def assign_fiber_addr(self, size):
        """Assign an address range for a fiber of given size"""
        addr = self.next_addr
        # Align to cache line boundary
        addr = (addr + self.fibercache.line_size_bytes - 1) // self.fibercache.line_size_bytes * self.fibercache.line_size_bytes
        # Reserve space (size elements * 16 bytes per element + padding)
        elem_size = 16  # 8 bytes for coordinate + 8 bytes for value
        self.next_addr = addr + size * elem_size + 64  # Add padding
        return addr
        
    def load_matrices(self, A, B):
        """
        Load matrices A and B for multiplication with proper FiberCache integration
        """
        self.A = A
        self.B = B
        
        # Assign IDs and addresses to all fibers in B
        for i, row in enumerate(B):
            # Assign ID if not already present
            if not hasattr(row, 'id'):
                row.id = self.assign_fiber_id()
            
            # Assign address if not already present
            if not hasattr(row, 'addr'):
                row.addr = self.assign_fiber_addr(len(row.coords))
                
            # Store fiber info
            self.fiber_info[row.id] = (len(row.coords), False)  # Not a partial result
                
            # Prefetch fiber into cache
            self.fibercache.fetch(row.id, row.addr, len(row.coords) * 16)
            
            # Manually update cache to populate it for simulation
            for j in range(len(row.coords)):
                elem_addr = row.addr + j * 16
                self.fibercache.write(elem_addr, (row.coords[j], row.values[j]))
        
        # Update FiberCache statistics
        self.fibercache.stats["B_fibers_stored"] = len(B)
            
    def run_simulation(self, max_cycles=1000000):
        """Run the cycle-accurate simulation"""
        done = False
        start_time = time.time()
        initial_memory_traffic = self.estimate_compulsory_traffic()
        
        while not done and self.cycle < max_cycles:
            # Update FiberCache (process pending fetches)
            self.fibercache.update_cycles(self.cycle)
            
            # Update PE status (check if any PEs have completed)
            self.scheduler.update_pe_status(self.pes)
            
            # Assign new tasks to idle PEs
            self.assign_tasks_to_pes()
            
            # Tick all PEs
            for pe_idx, pe in enumerate(self.pes):
                if pe is not None:
                    # Only tick PEs that are busy (have been assigned a task)
                    if self.scheduler.pe_status[pe_idx]:
                        pe.tick()
                        self.system_stats["active_pe_cycles"] += 1
                    else:
                        self.system_stats["idle_pe_cycles"] += 1
                    
            # Check if we're done
            done = self.scheduler.is_schedule_complete()
            
            # Increment cycle counter
            self.cycle += 1
            
            # Print progress every 10000 cycles
            if self.cycle % 10000 == 0:
                elapsed = time.time() - start_time
                print(f"Cycle {self.cycle}, elapsed time: {elapsed:.2f}s")
                
        print(f"\nSimulation completed after {self.cycle} cycles")
        
        # Calculate final statistics
        self.update_system_stats(initial_memory_traffic)
        self.print_stats()
        
    def assign_tasks_to_pes(self):
        """Assign tasks to idle PEs with proper FiberCache integration"""
        for pe_idx in range(self.num_pes):
            # Check if PE is idle
            if not self.scheduler.pe_status[pe_idx] and self.pes[pe_idx] is not None:
                if len(self.scheduler.task_queue) > 0:
                    task = self.scheduler.task_queue.popleft()
                    
                    # Process the task based on its type
                    if task["type"] == "leaf":
                        # Set input fibers with proper cache integration
                        input_fibers = task["B_fibers"]
                        fiber_ids = [fiber.id for fiber in input_fibers]
                        fiber_addrs = [fiber.addr for fiber in input_fibers]
                        
                        # Set output fiber information
                        output_fiber_id = self.assign_fiber_id()
                        output_fiber_addr = self.assign_fiber_addr(100)  # Estimate initial size
                        
                        # Register output fiber with scheduler
                        task["output_id"] = output_fiber_id
                        task["output_addr"] = output_fiber_addr
                        
                        # Set up the PE
                        self.pes[pe_idx].set_input_fibers(
                            fibers=input_fibers,
                            fiber_cache=self.fibercache,
                            fiber_ids=fiber_ids,
                            fiber_addrs=fiber_addrs
                        )
                        self.pes[pe_idx].output_fiber_id = output_fiber_id
                        self.pes[pe_idx].output_fiber_addr = output_fiber_addr
                        
                        # Mark PE as busy
                        self.scheduler.pe_status[pe_idx] = True
                        
                        # Track fiber assignments
                        for fiber_id in fiber_ids:
                            self.fiber_assignments[fiber_id] = pe_idx
                        self.fiber_assignments[output_fiber_id] = pe_idx
                        self.fiber_info[output_fiber_id] = (100, task["type"] != "root")  # Mark as partial result if not root
                        
                    elif task["type"] == "merge":
                        # Get input fibers (partial results)
                        child_tasks = task["children"]
                        input_fibers = []
                        fiber_ids = []
                        fiber_addrs = []
                        
                        for child in child_tasks:
                            # Create a fiber representation for this partial result
                            if "output_id" in child and "output_addr" in child:
                                # Get result fiber from the completed child task
                                fiber = Fiber([], [], scaling_factor=1.0)  # Empty fiber as placeholder
                                fiber.id = child["output_id"]
                                fiber.addr = child["output_addr"]
                                input_fibers.append(fiber)
                                fiber_ids.append(fiber.id)
                                fiber_addrs.append(fiber.addr)
                        
                        # Set output fiber information
                        output_fiber_id = self.assign_fiber_id()
                        output_fiber_addr = self.assign_fiber_addr(200)  # Estimate larger size for merge result
                        
                        # Register output fiber with scheduler
                        task["output_id"] = output_fiber_id
                        task["output_addr"] = output_fiber_addr
                        
                        # Set up the PE
                        self.pes[pe_idx].set_input_fibers(
                            fibers=input_fibers,
                            fiber_cache=self.fibercache,
                            fiber_ids=fiber_ids,
                            fiber_addrs=fiber_addrs
                        )
                        self.pes[pe_idx].output_fiber_id = output_fiber_id
                        self.pes[pe_idx].output_fiber_addr = output_fiber_addr
                        
                        # Mark PE as busy
                        self.scheduler.pe_status[pe_idx] = True
                        
                        # Track fiber assignments
                        for fiber_id in fiber_ids:
                            self.fiber_assignments[fiber_id] = pe_idx
                        self.fiber_assignments[output_fiber_id] = pe_idx
                        self.fiber_info[output_fiber_id] = (200, not task.get("is_root", False))  # Mark as partial result if not root
        
    def estimate_compulsory_traffic(self):
        """Estimate compulsory memory traffic (minimum required)"""
        # Calculate size of all input fibers
        input_size = 0
        for row in self.B:
            input_size += len(row.coords) * 16  # 16 bytes per element
            
        # Estimate size of output
        output_size = sum(len(row.coords) for row in self.A) * 16  # Very rough estimate
            
        return input_size + output_size
        
    def update_system_stats(self, initial_memory_traffic):
        """Update system-wide statistics before reporting"""
        # Collect total elements processed from PEs
        self.system_stats["total_elements_processed"] = sum(pe.stats["total_elements_processed"] for pe in self.pes if pe is not None)
        
        # Calculate PE utilization
        total_pe_cycles = self.system_stats["active_pe_cycles"] + self.system_stats["idle_pe_cycles"]
        if total_pe_cycles > 0:
            self.system_stats["pe_utilization"] = self.system_stats["active_pe_cycles"] / total_pe_cycles
            
        # Calculate memory bandwidth utilization
        total_memory_traffic = self.fibercache.stats["total_bandwidth_used"]
        if self.cycle > 0:
            average_bandwidth = total_memory_traffic / self.cycle
            self.system_stats["memory_bandwidth_utilization"] = average_bandwidth / self.memory_bw_limit
            
        # Calculate memory traffic reduction
        if initial_memory_traffic > 0:
            actual_traffic = self.fibercache.stats["total_bandwidth_used"]
            self.system_stats["memory_traffic_reduction"] = 1.0 - (actual_traffic / initial_memory_traffic)
        
    def print_stats(self):
        """Print statistics for the entire system"""
        print("\n=== GAMMA SYSTEM STATISTICS ===")
        print(f"Total cycles: {self.cycle}")
        
        # Print PE statistics
        print("\nPE Statistics:")
        active_pes = 0
        for i, pe in enumerate(self.pes):
            if pe is not None and pe.stats["total_elements_processed"] > 0:
                active_pes += 1
                print(f"PE {i}: {pe.cycle} cycles, {pe.stats['total_elements_processed']} elements processed")
                if pe.stats["cache_read_hits"] + pe.stats["cache_read_misses"] > 0:
                    hit_rate = pe.stats["cache_read_hits"] / (pe.stats["cache_read_hits"] + pe.stats["cache_read_misses"]) * 100
                    print(f"    Cache hit rate: {hit_rate:.2f}%")
        
        print(f"\nActive PEs: {active_pes}/{self.num_pes}")
        print(f"Total elements processed: {self.system_stats['total_elements_processed']}")
        print(f"PE utilization: {self.system_stats['pe_utilization'] * 100:.2f}%")
        print(f"Memory bandwidth utilization: {self.system_stats['memory_bandwidth_utilization'] * 100:.2f}%")
        
        if self.system_stats["memory_traffic_reduction"] > 0:
            print(f"Memory traffic reduction: {self.system_stats['memory_traffic_reduction'] * 100:.2f}%")
        
        # Allow components to print their own stats
        self.fibercache.print_stats()
        self.scheduler.print_stats()


def create_test_matrices():
    """Create test matrices to verify the system"""
    # Create test matrix A
    a_rows = []
    for i in range(10):
        coords = [j for j in range(5) if random.random() < 0.3]
        values = [random.random() for _ in range(len(coords))]
        fiber = Fiber(coords, values, 1.0)
        # Add index attribute to the fiber for tracking
        fiber.index = i
        a_rows.append(fiber)
        
    # Create test matrix B
    b_rows = []
    for i in range(5):
        coords = [j for j in range(10) if random.random() < 0.3]
        values = [random.random() for _ in range(len(coords))]
        fiber = Fiber(coords, values, 1.0)
        # Add index attribute to the fiber for tracking
        fiber.index = i
        b_rows.append(fiber)
        
    return a_rows, b_rows


def run_simple_test_case():
    """Run a simple test case similar to the paper example"""
    print("=== RUNNING SIMPLE TEST CASE ===")
    
    # Create fibers from the paper example
    # B3 from paper example: column coordinates 2 and 4 with values
    b3 = Fiber(coords=[2, 4], values=[0.2, 0.4], scaling_factor=0.3)  # a_1,3 = 0.3
    b3.index = 3
    
    # B5 from paper example: column coordinates 1 and 4 with values
    b5 = Fiber(coords=[1, 4], values=[0.5, 0.6], scaling_factor=0.5)  # a_1,5 = 0.5
    b5.index = 5
    
    # Create A1 row with nonzeros at indices 3 and 5
    a1 = Fiber(coords=[3, 5], values=[0.3, 0.5], scaling_factor=1.0)
    a1.index = 1
    
    # Initialize Gamma system with just 1 PE for simplicity
    gamma = GammaSystem(num_pes=1, pe_radix=64, cache_size_mb=3)
    
    # Setup the test manually
    gamma.pes[0].set_input_fibers([b3, b5])
    
    # Run simulation for a few cycles
    max_cycles = 50
    cycles_taken = 0
    
    for _ in range(max_cycles):
        # Tick the PE
        done = gamma.pes[0].tick()
        cycles_taken += 1
        
        if done:
            # Add one more cycle to finish
            cycles_taken += 1
            break
    
    # Get result
    result = gamma.pes[0].get_result_fiber()
    
    # Print results
    print("\nInput fibers:")
    print(f"B3 (scaled by 0.3): coords={b3.coords}, values={b3.values}")
    print(f"B5 (scaled by 0.5): coords={b5.coords}, values={b5.values}")
    
    print("\nExpected output fiber:")
    # Expected calculations:
    # coord 1: 0.5 * 0.5 = 0.25
    # coord 2: 0.2 * 0.3 = 0.06
    # coord 4: 0.4 * 0.3 + 0.6 * 0.5 = 0.12 + 0.3 = 0.42
    print("coords=[1, 2, 4], values=[0.25, 0.06, 0.42]")
    
    print("\nActual output fiber:")
    print(f"coords={result.coords}, values={result.values}")
    print(f"\nSimulation took {cycles_taken} cycles")
    
    # Validate results
    expected_coords = [1, 2, 4]
    expected_values = [0.25, 0.06, 0.42]
    
    for i in range(len(expected_coords)):
        if abs(result.values[i] - expected_values[i]) > 1e-10:
            print(f"\nTest FAILED at index {i}")
            return False
    
    print("\nTest PASSED: Result matches expected output")
    gamma.pes[0].print_stats()
    return True

def run_multi_pe_test():
    """Test the full Gamma system with multiple PEs"""
    print("\n=== TESTING FULL GAMMA SYSTEM WITH MULTIPLE PES ===")
    
    # Create test matrices
    A, B = create_test_matrices()
    
    # Initialize Gamma system with a small number of PEs for quick testing
    gamma = GammaSystem(num_pes=4, pe_radix=64, cache_size_mb=3)
    
    # Load matrices
    gamma.load_matrices(A, B)
    
    # Schedule some rows manually
    for i in range(min(3, len(A))):
        B_rows = [B[j] for j in A[i].coords if j < len(B)]
        if B_rows:
            gamma.scheduler.schedule_row(A[i], B_rows)
    
    # Run simulation
    gamma.run_simulation(max_cycles=1000)
    
    print("Multi-PE test completed successfully")

def run_gamma_test():
    """Test the full Gamma system"""
    print("=== TESTING FULL GAMMA SYSTEM ===")
    
    # Run single PE test first
    run_simple_test_case()
    
    # Run multi-PE test
    run_multi_pe_test()
    
    print("All tests completed successfully")


if __name__ == "__main__":
    run_gamma_test()