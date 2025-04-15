"""
Cycle-Accurate Simulator for Gamma Processing Element

This simulator models the Gamma PE from the paper:
"Gamma: Leveraging Gustavson's Algorithm to Accelerate Sparse Matrix Multiplication"
"""

class Fiber:
    """Represents a sparse fiber (row or column) with coordinates and values"""
    def __init__(self, coords, values, scaling_factor=1.0):
        # Ensure coordinates and values are sorted by coordinate
        assert len(coords) == len(values), "Coordinates and values must have the same length"
        assert all(coords[i] <= coords[i+1] for i in range(len(coords)-1)), "Coordinates must be sorted"
        
        self.coords = list(coords)  # Column/row coordinates
        self.values = list(values)  # Nonzero values
        self.scaling_factor = scaling_factor  # Scaling factor (a_mk for B rows)
        self.size = len(coords)
    
    def __repr__(self):
        return f"Fiber(size={self.size}, scaling={self.scaling_factor})"


class FiberBuffer:
    """Represents input buffers in the PE for coordinates and values"""
    def __init__(self, size=16):  # Default small buffer size
        self.coords = []
        self.values = []
        self.capacity = size
    
    def is_empty(self):
        return len(self.coords) == 0
    
    def is_full(self):
        return len(self.coords) >= self.capacity
    
    def enqueue(self, coord, value):
        """Add element to buffer"""
        if self.is_full():
            return False
        self.coords.append(coord)
        self.values.append(value)
        return True
    
    def dequeue(self):
        """Remove and return front element"""
        if self.is_empty():
            return None, None
        return self.coords.pop(0), self.values.pop(0)
    
    def peek(self):
        """Look at front element without removing"""
        if self.is_empty():
            return None, None
        return self.coords[0], self.values[0]


class GammaPE:
    """Cycle-accurate model of the Gamma Processing Element with FiberCache integration"""
    def __init__(self, radix=64):
        self.radix = radix  # Number of input fibers that can be combined
        self.cycle = 0      # Current cycle count
        
        # Pipeline stages
        self.fetch_stage = None
        self.merge_stage = None
        self.multiply_stage = None
        self.accumulate_stage = None
        
        # Input buffers (one for each input fiber)
        self.input_buffers = [FiberBuffer() for _ in range(radix)]
        
        # Output buffer for the result fiber
        self.output_fiber = Fiber([], [])
        
        # Accumulator for values with the same coordinate
        self.accumulator = {}  # Maps coordinates to accumulated values
        self.current_coord = None  # Current coordinate being accumulated
        
        # Scaling factors for each input fiber
        self.scaling_factors = [1.0] * radix
        
        # Input fibers and read positions
        self.input_fibers = [None] * radix
        self.read_positions = [0] * radix
        
        # Pipeline stall signals
        self.fetch_stall = False
        self.merge_stall = False
        self.multiply_stall = False
        self.accumulate_stall = False
        
        # Integration with FiberCache
        self.fiber_cache = None  # Reference to FiberCache
        self.fiber_ids = [None] * radix  # IDs of input fibers
        self.fiber_addrs = [None] * radix  # Base addresses of input fibers
        self.output_fiber_id = None  # ID of output fiber
        self.output_fiber_addr = None  # Base address of output fiber
        self.pending_reads = []  # List of pending reads from cache
        self.read_stall = False  # Stall due to cache miss
        
        # Debug stats
        self.stats = {
            "fetch_cycles": 0,
            "merge_cycles": 0,
            "multiply_cycles": 0,
            "accumulate_cycles": 0,
            "stall_cycles": 0,
            "cache_read_hits": 0,
            "cache_read_misses": 0,
            "cache_write_hits": 0,
            "cache_write_misses": 0,
            "total_elements_processed": 0
        }
    
    def reset(self):
        """Reset the PE state"""
        self.cycle = 0
        self.fetch_stage = None
        self.merge_stage = None
        self.multiply_stage = None
        self.accumulate_stage = None
        self.current_coord = None
        self.accumulator = {}
        self.output_fiber = Fiber([], [])
        self.read_positions = [0] * self.radix
        self.fetch_stall = False
        self.merge_stall = False
        self.multiply_stall = False
        self.accumulate_stall = False
        self.read_stall = False
        self.pending_reads = []
        
        # Reset stats
        for key in self.stats:
            self.stats[key] = 0
    
    def set_input_fibers(self, fibers, fiber_cache=None, fiber_ids=None, fiber_addrs=None):
        """
        Set the input fibers for processing
        
        Args:
            fibers: List of input fibers
            fiber_cache: Reference to FiberCache
            fiber_ids: List of fiber IDs (optional)
            fiber_addrs: List of fiber base addresses (optional)
        """
        assert len(fibers) <= self.radix, f"Cannot process more than {self.radix} fibers at once"
        
        # Reset the PE state
        self.reset()
        
        # Store input fibers and their scaling factors
        for i, fiber in enumerate(fibers):
            if fiber is not None:
                self.input_fibers[i] = fiber
                self.scaling_factors[i] = fiber.scaling_factor
            else:
                self.input_fibers[i] = None
        
        # Fill remaining positions with None
        for i in range(len(fibers), self.radix):
            self.input_fibers[i] = None
        
        # Initialize read positions
        self.read_positions = [0 if f is not None else -1 for f in self.input_fibers]
        
        # Set FiberCache integration properties
        self.fiber_cache = fiber_cache
        
        # Set fiber IDs if provided
        if fiber_ids is not None:
            self.fiber_ids = fiber_ids.copy()
            while len(self.fiber_ids) < self.radix:
                self.fiber_ids.append(None)
        else:
            # Try to extract IDs from fibers if they have them
            self.fiber_ids = [getattr(f, 'id', None) for f in fibers]
            while len(self.fiber_ids) < self.radix:
                self.fiber_ids.append(None)
        
        # Set fiber addresses if provided
        if fiber_addrs is not None:
            self.fiber_addrs = fiber_addrs.copy()
            while len(self.fiber_addrs) < self.radix:
                self.fiber_addrs.append(None)
        else:
            # Try to extract addresses from fibers if they have them
            self.fiber_addrs = [getattr(f, 'addr', None) for f in fibers]
            while len(self.fiber_addrs) < self.radix:
                self.fiber_addrs.append(None)
    
    def fetch_stage_logic(self):
        """Fetching elements from input fibers to buffers"""
        # If we're stalled due to cache miss, check if reads are complete
        if self.read_stall and self.fiber_cache:
            # Check if any pending reads are still in progress
            still_pending = False
            for fiber_idx, pos in self.pending_reads:
                # Calculate address for this element
                if self.fiber_addrs[fiber_idx] is not None:
                    elem_addr = self.fiber_addrs[fiber_idx] + pos * 16  # Assuming 16 bytes per element (coord + value)
                    # Check if element is in cache now
                    if self.fiber_cache._find_line(elem_addr)[0] is not None:
                        # Element is now in cache, can proceed
                        continue
                    else:
                        # Still waiting for at least one element
                        still_pending = True
                        break
            
            # If all reads are complete, clear stall flag
            if not still_pending:
                self.read_stall = False
                self.pending_reads = []
        
        # Skip if we're still stalled
        if self.read_stall:
            self.stats["stall_cycles"] += 1
            return False
        
        # Check if we need to fetch more elements
        fetched = False
        self.pending_reads = []  # Clear pending reads list
        
        for i in range(self.radix):
            # Skip if fiber is empty or finished
            if (self.input_fibers[i] is None or 
                self.read_positions[i] >= self.input_fibers[i].size or
                self.input_buffers[i].is_full()):
                continue
            
            # Fetch next element
            pos = self.read_positions[i]
            fiber = self.input_fibers[i]
            
            # If we have a FiberCache and fiber address, use cache
            cache_hit = False
            if self.fiber_cache and self.fiber_addrs[i] is not None:
                # Calculate address for this element
                elem_addr = self.fiber_addrs[i] + pos * 16  # Assuming 16 bytes per element
                
                # Attempt to read from cache
                cache_data = self.fiber_cache.read(elem_addr)
                
                if cache_data is not None:
                    # Cache hit
                    self.stats["cache_read_hits"] += 1
                    cache_hit = True
                    
                    # In a real system, cache_data would contain the element
                    # For simplicity in simulation, we'll still use the fiber's data
                    coord = fiber.coords[pos]
                    value = fiber.values[pos]
                else:
                    # Cache miss
                    self.stats["cache_read_misses"] += 1
                    
                    # In a real system, this would trigger a fetch and stall
                    # For simulation, we'll track it but proceed with direct access
                    # Record this as a pending read
                    self.pending_reads.append((i, pos))
                    
                    # In a complete simulator, we would stall here
                    # self.read_stall = True
                    # return False
                    
                    # For now, just access directly
                    coord = fiber.coords[pos]
                    value = fiber.values[pos]
            else:
                # Direct access (no cache)
                coord = fiber.coords[pos]
                value = fiber.values[pos]
            
            # Add element to buffer
            if self.input_buffers[i].enqueue(coord, value):
                self.read_positions[i] += 1
                fetched = True
        
        # Update fetch stage output
        if fetched:
            self.fetch_stage = "Fetched"
            self.stats["fetch_cycles"] += 1
        else:
            self.fetch_stage = None
        
        # Check if all fibers are processed
        all_processed = all(
            self.input_fibers[i] is None or 
            (self.read_positions[i] >= self.input_fibers[i].size and self.input_buffers[i].is_empty())
            for i in range(self.radix)
        )
        
        return all_processed and not fetched
    
    def merge_stage_logic(self):
        """Find the minimum coordinate among buffer heads"""
        if self.merge_stall:
            self.stats["stall_cycles"] += 1
            return False
        
        min_coord = float('inf')
        min_way = -1
        
        # Find the smallest coordinate across all input buffers
        for i in range(self.radix):
            coord, _ = self.input_buffers[i].peek()
            if coord is not None and coord < min_coord:
                min_coord = coord
                min_way = i
        
        # If we found a valid minimum coordinate
        if min_way != -1:
            coord, value = self.input_buffers[min_way].dequeue()
            self.merge_stage = (min_way, coord, value)
            self.stats["merge_cycles"] += 1
            return False
        else:
            self.merge_stage = None
            return True
    
    def multiply_stage_logic(self):
        """Scale the value by the appropriate scaling factor"""
        if self.merge_stage is None:
            self.multiply_stage = None
            return True
        
        if self.multiply_stall:
            self.stats["stall_cycles"] += 1
            return False
        
        way, coord, value = self.merge_stage
        scaled_value = value * self.scaling_factors[way]
        
        self.multiply_stage = (coord, scaled_value)
        self.stats["multiply_cycles"] += 1
        return False
    
    def accumulate_stage_logic(self):
        """Accumulate values with the same coordinate"""
        if self.multiply_stage is None:
            self.accumulate_stage = None
            # If no more elements are coming but we have a value in accumulator,
            # emit it now
            if self.current_coord is not None:
                # If we have a FiberCache and output address, write to cache
                if self.fiber_cache and self.output_fiber_addr is not None:
                    # Calculate address for this output element
                    elem_addr = self.output_fiber_addr + len(self.output_fiber.coords) * 16
                    
                    # Write to cache
                    self.fiber_cache.write(
                        elem_addr, 
                        (self.current_coord, self.accumulator[self.current_coord]),
                        is_partial_result=(self.output_fiber_id is not None)
                    )
                
                # Add to output fiber
                self.output_fiber.coords.append(self.current_coord)
                self.output_fiber.values.append(self.accumulator[self.current_coord])
                self.current_coord = None
            return True
        
        if self.accumulate_stall:
            self.stats["stall_cycles"] += 1
            return False
        
        coord, value = self.multiply_stage
        
        # If this is a new coordinate and we have accumulated a previous value, emit it
        if self.current_coord is not None and coord != self.current_coord:
            # If we have a FiberCache and output address, write to cache
            if self.fiber_cache and self.output_fiber_addr is not None:
                # Calculate address for this output element
                elem_addr = self.output_fiber_addr + len(self.output_fiber.coords) * 16
                
                # Write to cache
                self.fiber_cache.write(
                    elem_addr, 
                    (self.current_coord, self.accumulator[self.current_coord]),
                    is_partial_result=(self.output_fiber_id is not None)
                )
            
            # Add to output fiber
            self.output_fiber.coords.append(self.current_coord)
            self.output_fiber.values.append(self.accumulator[self.current_coord])
        
        # Start or continue accumulating current coordinate
        if coord not in self.accumulator:
            self.accumulator[coord] = 0
        
        self.accumulator[coord] += value
        self.current_coord = coord
        
        self.accumulate_stage = coord
        self.stats["accumulate_cycles"] += 1
        self.stats["total_elements_processed"] += 1
        return False
    
    def tick(self):
        """Advance simulation by one cycle"""
        # Process pipeline stages from back to front to avoid overwriting data
        accumulate_done = self.accumulate_stage_logic()
        multiply_done = self.multiply_stage_logic()
        merge_done = self.merge_stage_logic()
        fetch_done = self.fetch_stage_logic()
        
        # Advance simulation cycle
        self.cycle += 1
        
        # Return True if all stages are done
        return fetch_done and merge_done and multiply_done and accumulate_done
    
    def run(self, max_cycles=10000):
        """Run the PE simulation until completion or max cycles"""
        done = False
        while not done and self.cycle < max_cycles:
            done = self.tick()
            
            # Increment a final cycle count to drain the pipeline
            if done:
                self.cycle += 1
        
        # Update output fiber size
        self.output_fiber.size = len(self.output_fiber.coords)
        return self.cycle
    
    def get_result_fiber(self):
        """Return the result fiber after processing"""
        return self.output_fiber
    
    def print_stats(self):
        """Print simulation statistics"""
        print(f"\nSimulation completed in {self.cycle} cycles")
        print(f"Fetch cycles: {self.stats['fetch_cycles']}")
        print(f"Merge cycles: {self.stats['merge_cycles']}")
        print(f"Multiply cycles: {self.stats['multiply_cycles']}")
        print(f"Accumulate cycles: {self.stats['accumulate_cycles']}")
        print(f"Stall cycles: {self.stats['stall_cycles']}")
        print(f"Cache read hits: {self.stats['cache_read_hits']}")
        print(f"Cache read misses: {self.stats['cache_read_misses']}")
        print(f"Cache hit rate: {self.stats['cache_read_hits'] / max(1, self.stats['cache_read_hits'] + self.stats['cache_read_misses']) * 100:.2f}%")
        print(f"Elements processed: {self.stats['total_elements_processed']}")
        print(f"Output elements: {self.output_fiber.size}")


def run_simulation_test_case():
    """Run a test simulation of the Gamma PE"""
    print("=== GAMMA PE CYCLE-ACCURATE SIMULATOR ===")
    
    # Create test fibers
    # B3 from paper example: column coordinates 2 and 4 with values
    b3 = Fiber(coords=[2, 4], values=[0.2, 0.4], scaling_factor=0.3)  # a_1,3 = 0.3
    
    # B5 from paper example: column coordinates 1 and 4 with values
    b5 = Fiber(coords=[1, 4], values=[0.5, 0.6], scaling_factor=0.5)  # a_1,5 = 0.5
    
    # Initialize the PE
    pe = GammaPE(radix=64)
    
    # Set input fibers
    pe.set_input_fibers([b3, b5])
    
    # Run simulation
    cycles = pe.run()
    
    # Get result
    result = pe.get_result_fiber()
    
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
    
    # Validate results
    expected_coords = [1, 2, 4]
    expected_values = [0.25, 0.06, 0.42]
    
    if len(result.coords) != len(expected_coords):
        print("\nTest FAILED: Output size mismatch")
        return False
    
    for i in range(len(expected_coords)):
        if result.coords[i] != expected_coords[i] or abs(result.values[i] - expected_values[i]) > 1e-10:
            print(f"\nTest FAILED at index {i}")
            return False
    
    print("\nTest PASSED: Result matches expected output")
    pe.print_stats()
    return True


def run_complex_test_case():
    """Run a more complex test case with more fibers and coordinates"""
    print("\n=== COMPLEX TEST CASE ===")
    
    # Create larger test fibers
    b1 = Fiber(coords=[0, 3, 5, 7], values=[1.1, 1.3, 1.5, 1.7], scaling_factor=0.1)
    b2 = Fiber(coords=[1, 3, 6, 8], values=[2.1, 2.3, 2.6, 2.8], scaling_factor=0.2)
    b3 = Fiber(coords=[2, 5, 7, 9], values=[3.2, 3.5, 3.7, 3.9], scaling_factor=0.3)
    b4 = Fiber(coords=[0, 4, 6, 9], values=[4.0, 4.4, 4.6, 4.9], scaling_factor=0.4)
    
    # Initialize the PE
    pe = GammaPE(radix=64)
    
    # Set input fibers
    pe.set_input_fibers([b1, b2, b3, b4])
    
    # Run simulation
    cycles = pe.run()
    
    # Get result
    result = pe.get_result_fiber()
    
    # Manually calculate expected values
    expected = {}
    
    # Process b1
    for i in range(len(b1.coords)):
        c, v = b1.coords[i], b1.values[i]
        if c not in expected:
            expected[c] = 0
        expected[c] += v * b1.scaling_factor
    
    # Process b2
    for i in range(len(b2.coords)):
        c, v = b2.coords[i], b2.values[i]
        if c not in expected:
            expected[c] = 0
        expected[c] += v * b2.scaling_factor
    
    # Process b3
    for i in range(len(b3.coords)):
        c, v = b3.coords[i], b3.values[i]
        if c not in expected:
            expected[c] = 0
        expected[c] += v * b3.scaling_factor
    
    # Process b4
    for i in range(len(b4.coords)):
        c, v = b4.coords[i], b4.values[i]
        if c not in expected:
            expected[c] = 0
        expected[c] += v * b4.scaling_factor
    
    # Convert to sorted lists
    expected_coords = sorted(expected.keys())
    expected_values = [expected[c] for c in expected_coords]
    
    # Print results
    print(f"\nExpected output fiber:")
    print(f"coords={expected_coords}")
    print(f"values={expected_values}")
    
    print(f"\nActual output fiber:")
    print(f"coords={result.coords}")
    print(f"values={result.values}")
    
    # Validate results
    if len(result.coords) != len(expected_coords):
        print("\nTest FAILED: Output size mismatch")
        return False
    
    for i in range(len(expected_coords)):
        if result.coords[i] != expected_coords[i] or abs(result.values[i] - expected_values[i]) > 1e-10:
            print(f"\nTest FAILED at index {i}")
            return False
    
    print("\nTest PASSED: Complex test case result matches expected output")
    pe.print_stats()
    return True


def test_256_nonzeros_case():
    """Test case for a row with 256 nonzeros to match earlier calculation"""
    print("\n=== 256 NONZEROS TEST CASE ===")
    
    # Create 4 fibers with 64 nonzeros each
    fibers = []
    for f_idx in range(4):
        coords = []
        values = []
        for i in range(64):
            # Create unique coordinates across fibers
            coord = f_idx * 100 + i
            value = (f_idx + 1) * 0.1 + i * 0.01
            coords.append(coord)
            values.append(value)
        fibers.append(Fiber(coords=coords, values=values, scaling_factor=1.0))
    
    # Initialize the PE
    pe = GammaPE(radix=64)
    
    # Process each fiber and record cycles
    total_cycles = 0
    partial_results = []
    
    # Process first level
    for i in range(4):
        pe.reset()
        pe.set_input_fibers([fibers[i]])
        cycles = pe.run()
        total_cycles += cycles
        partial_results.append(pe.get_result_fiber())
        print(f"Processing fiber {i+1} took {cycles} cycles, produced {len(pe.get_result_fiber().coords)} elements")
    
    # Process second level - combine the 4 partial results
    pe.reset()
    pe.set_input_fibers(partial_results)
    cycles = pe.run()
    total_cycles += cycles
    
    final_result = pe.get_result_fiber()
    print(f"Combining 4 partial results took {cycles} cycles, produced {len(final_result.coords)} elements")
    print(f"Total processing cycles: {total_cycles}")
    
    # Since all coordinates are unique, we should have 256 elements in the final result
    assert len(final_result.coords) == 256, "Expected 256 elements in final result"
    
    pe.print_stats()
    return True


if __name__ == "__main__":
    run_simulation_test_case()
    run_complex_test_case()
    test_256_nonzeros_case()