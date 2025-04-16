class FiberCache:
    """
    FiberCache - A specialized cache structure for the Gamma accelerator that combines
    features of both traditional caches and explicitly managed buffers to efficiently
    capture irregular reuse patterns of sparse fibers.
    """
    def __init__(self, total_size_bytes=3*1024*1024, line_size=64, num_banks=48, associativity=16, memory_system=None):
        """
        Initialize the FiberCache.
        
        Args:
            total_size_bytes: Total size of the cache in bytes (default 3MB)
            line_size: Size of each cache line in bytes
            num_banks: Number of banks for parallel access
            associativity: Number of ways in each set
            memory_system: Reference to the memory system
        """
        self.total_size_bytes = total_size_bytes
        self.line_size = line_size
        self.num_banks = num_banks
        self.associativity = associativity
        self.memory = memory_system
        
        # Calculate number of sets per bank
        self.total_lines = total_size_bytes // line_size
        self.sets_per_bank = self.total_lines // (associativity * num_banks)
        
        # Priority counter parameters (5-bit counter as mentioned in the paper)
        self.MAX_PRIORITY = 31  # 2^5 - 1
        
        # RRIP parameters (2-bit as mentioned in the paper)
        self.M = 2  # Width of RRPV register
        self.MAX_RRPV = (1 << self.M) - 1  # Maximum RRPV value (3 for 2-bit)
        self.LONG_RRPV = self.MAX_RRPV - 1  # Long re-reference interval (2 for 2-bit)
        
        # Initialize the cache structure
        # Each entry contains: [valid, dirty, tag, priority, rrpv, data]
        self.cache = [[[False, False, 0, 0, self.MAX_RRPV, bytearray(line_size)] 
                      for _ in range(associativity)] 
                      for _ in range(self.sets_per_bank * num_banks)]
        
        # Track ongoing bank operations
        self.bank_busy_until = [0] * num_banks
        
        # Outstanding fetch requests
        self.outstanding_fetches = {}  # address -> completion_cycle
        
        # Statistics
        self.stats = {
            'fetch_requests': 0,
            'read_requests': 0,
            'write_requests': 0,
            'consume_requests': 0,
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'writebacks': 0,
            'bank_conflicts': 0,
            'current_cycle': 0
        }
    
    def _get_bank_set_way(self, address):
        """
        Convert an address to bank, set, and tag.
        
        Args:
            address: Memory address
            
        Returns:
            Tuple of (bank, set_within_bank, tag, set_index_global)
        """
        line_address = address // self.line_size
        bank = line_address % self.num_banks
        set_within_bank = (line_address // self.num_banks) % self.sets_per_bank
        set_index_global = bank * self.sets_per_bank + set_within_bank
        tag = line_address // (self.num_banks * self.sets_per_bank)
        
        return bank, set_within_bank, tag, set_index_global
    
    def _find_line(self, set_index, tag):
        """
        Find a line in a set with the given tag.
        
        Args:
            set_index: Index of the set
            tag: Tag to look for
            
        Returns:
            Way index if found, None otherwise
        """
        for way in range(self.associativity):
            if self.cache[set_index][way][0] and self.cache[set_index][way][2] == tag:
                return way
        return None
    
    def _find_victim(self, set_index):
        """
        Find a victim line in the set according to RRIP replacement policy.
        
        Args:
            set_index: Index of the set
            
        Returns:
            Way index of the victim
        """
        # First, try to find an invalid line
        for way in range(self.associativity):
            if not self.cache[set_index][way][0]:
                return way
        
        # Next, look for a line with distant re-reference interval (MAX_RRPV)
        for way in range(self.associativity):
            if self.cache[set_index][way][4] == self.MAX_RRPV:
                return way
        
        # If no line has MAX_RRPV, increment all RRPVs and try again
        for way in range(self.associativity):
            self.cache[set_index][way][4] = min(self.MAX_RRPV, self.cache[set_index][way][4] + 1)
        
        # Now try again to find a line with MAX_RRPV
        return self._find_victim(set_index)
    
    def _is_bank_busy(self, bank, current_cycle):
        """
        Check if a bank is busy.
        
        Args:
            bank: Bank number
            current_cycle: Current cycle
            
        Returns:
            Boolean indicating if bank is busy and when it will be free
        """
        return self.bank_busy_until[bank] > current_cycle, self.bank_busy_until[bank]
    
    def _set_bank_busy(self, bank, cycles, current_cycle):
        """
        Mark a bank as busy for a number of cycles.
        
        Args:
            bank: Bank number
            cycles: Number of cycles the operation will take
            current_cycle: Current cycle
        """
        self.bank_busy_until[bank] = max(self.bank_busy_until[bank], current_cycle + cycles)
    
    def _writeback_line(self, set_index, way, callback=None):
        """
        Write back a dirty cache line to memory.
        
        Args:
            set_index: Set index
            way: Way index
            callback: Function to call when writeback completes
            
        Returns:
            Boolean indicating if writeback was initiated
        """
        if not self.cache[set_index][way][0] or not self.cache[set_index][way][1]:
            # Line is not valid or not dirty
            if callback:
                callback()
            return True
        
        # Calculate address from tag and set
        tag = self.cache[set_index][way][2]
        bank, set_within_bank = divmod(set_index, self.sets_per_bank)
        line_address = (tag * self.num_banks * self.sets_per_bank) + (bank * self.sets_per_bank) + set_within_bank
        address = line_address * self.line_size
        
        # Get data
        data = self.cache[set_index][way][5]
        
        # Write back to memory
        if self.memory:
            self.memory.write(address, data, callback)
            self.stats['writebacks'] += 1
            return True
        else:
            # No memory system attached
            if callback:
                callback()
            return False
    
    def fetch(self, address, size, pe_id=None):
        """
        Explicitly fetch data into the cache ahead of time.
        This is part of the decoupled data orchestration strategy.
        
        Args:
            address: Starting address of the data to fetch
            size: Size of the data in bytes
            pe_id: Processing Element ID requesting the fetch
            
        Returns:
            Boolean indicating success
        """
        self.stats['fetch_requests'] += 1
        current_cycle = self.stats['current_cycle']
        
        # Calculate number of cache lines needed
        start_line = address // self.line_size
        end_line = (address + size - 1) // self.line_size
        num_lines = end_line - start_line + 1
        
        # Fetch each line
        success = True
        current_address = start_line * self.line_size
        
        for _ in range(num_lines):
            bank, set_within_bank, tag, set_index = self._get_bank_set_way(current_address)
            
            # Check if bank is busy
            bank_busy, ready_cycle = self._is_bank_busy(bank, current_cycle)
            if bank_busy:
                self.stats['bank_conflicts'] += 1
            
            # Check if line is already in cache
            way = self._find_line(set_index, tag)
            
            if way is not None:
                # Line is already in cache - increase priority (max 5 bits)
                self.cache[set_index][way][3] = min(self.MAX_PRIORITY, self.cache[set_index][way][3] + 1)
                # Also update RRPV to indicate shorter re-reference interval
                self.cache[set_index][way][4] = 0  # Near-immediate re-reference
            else:
                # Line is not in cache - need to fetch it
                way = self._find_victim(set_index)
                
                # If victim is dirty, write it back to memory
                if self.cache[set_index][way][0] and self.cache[set_index][way][1]:
                    self._writeback_line(set_index, way)
                
                # Mark the bank as busy for this operation
                self._set_bank_busy(bank, 1, max(current_cycle, ready_cycle))
                
                # Fetch the line from memory
                if self.memory:
                    # In a real implementation, we would handle the asynchronous nature of this
                    def install_fetched_data(data):
                        # Install the new line in the cache with long re-reference interval
                        self.cache[set_index][way] = [
                            True,                   # valid
                            False,                  # dirty
                            tag,                    # tag
                            1,                      # priority (initial)
                            self.LONG_RRPV,         # RRPV (long interval)
                            data                    # data
                        ]
                    
                    self.memory.read(current_address, self.line_size, install_fetched_data)
                    self.outstanding_fetches[current_address] = ready_cycle + 1  # +1 for the operation
                else:
                    # No memory system - create dummy data
                    self.cache[set_index][way] = [
                        True,                   # valid
                        False,                  # dirty
                        tag,                    # tag
                        1,                      # priority (initial)
                        self.LONG_RRPV,         # RRPV (long interval)
                        bytearray(self.line_size)  # dummy data
                    ]
                
                self.stats['misses'] += 1
            
            current_address += self.line_size
        
        return success
    
    def read(self, address, size, pe_id=None):
        """
        Read data from the cache. This assumes the data has already been fetched.
        
        Args:
            address: Starting address of the data to read
            size: Size of the data in bytes
            pe_id: Processing Element ID requesting the read
            
        Returns:
            Data read from cache or None if not present
        """
        self.stats['read_requests'] += 1
        current_cycle = self.stats['current_cycle']
        
        # Calculate line address and offset
        line_address = address // self.line_size
        offset = address % self.line_size
        
        bank, set_within_bank, tag, set_index = self._get_bank_set_way(line_address * self.line_size)
        
        # Check if bank is busy
        bank_busy, ready_cycle = self._is_bank_busy(bank, current_cycle)
        if bank_busy:
            self.stats['bank_conflicts'] += 1
            # In a more detailed model, we would stall here
        
        # Mark the bank as busy for this operation
        self._set_bank_busy(bank, 1, max(current_cycle, ready_cycle))
        
        # Find the line in the cache
        way = self._find_line(set_index, tag)
        
        if way is not None:
            # Cache hit - update RRPV to near-immediate
            self.cache[set_index][way][4] = 0
            # Decrement priority (prevent overflow)
            self.cache[set_index][way][3] = max(0, self.cache[set_index][way][3] - 1)
            
            self.stats['hits'] += 1
            
            # Return the requested data
            if offset + size <= self.line_size:
                return bytes(self.cache[set_index][way][5][offset:offset+size])
            else:
                # Handle case where data spans multiple cache lines
                result = bytearray()
                result.extend(self.cache[set_index][way][5][offset:])
                
                remaining_size = size - (self.line_size - offset)
                current_address = (line_address + 1) * self.line_size
                
                while remaining_size > 0:
                    bank, set_within_bank, tag, set_index = self._get_bank_set_way(current_address)
                    
                    # Check if bank is busy
                    bank_busy, ready_cycle = self._is_bank_busy(bank, current_cycle)
                    if bank_busy:
                        self.stats['bank_conflicts'] += 1
                    
                    # Mark the bank as busy for this operation
                    self._set_bank_busy(bank, 1, max(current_cycle, ready_cycle))
                    
                    way = self._find_line(set_index, tag)
                    
                    if way is None:
                        # This should not happen as data should have been pre-fetched
                        self.stats['misses'] += 1
                        return None
                    
                    # Update RRPV to near-immediate
                    self.cache[set_index][way][4] = 0
                    # Decrement priority
                    self.cache[set_index][way][3] = max(0, self.cache[set_index][way][3] - 1)
                    
                    bytes_to_read = min(remaining_size, self.line_size)
                    result.extend(self.cache[set_index][way][5][:bytes_to_read])
                    
                    remaining_size -= bytes_to_read
                    current_address += self.line_size
                
                return bytes(result)
        else:
            # Cache miss - this shouldn't happen if fetch was called properly
            self.stats['misses'] += 1
            
            # Check if this is an outstanding fetch
            if address in self.outstanding_fetches:
                # The fetch is still pending - in a real implementation, would stall until complete
                pass
            
            return None
    
    def write(self, address, data, pe_id=None):
        """
        Write data to the cache without fetching from memory first.
        This is used for partial output fibers.
        
        Args:
            address: Starting address where to write
            data: Data to write
            pe_id: Processing Element ID performing the write
            
        Returns:
            Boolean indicating success
        """
        self.stats['write_requests'] += 1
        current_cycle = self.stats['current_cycle']
        
        # Calculate line address and offset
        line_address = address // self.line_size
        offset = address % self.line_size
        size = len(data)
        
        bank, set_within_bank, tag, set_index = self._get_bank_set_way(line_address * self.line_size)
        
        # Check if bank is busy
        bank_busy, ready_cycle = self._is_bank_busy(bank, current_cycle)
        if bank_busy:
            self.stats['bank_conflicts'] += 1
        
        # Mark the bank as busy for this operation
        self._set_bank_busy(bank, 1, max(current_cycle, ready_cycle))
        
        # Find or allocate a line in the cache
        way = self._find_line(set_index, tag)
        
        if way is None:
            # Allocate a new line without fetching from memory
            way = self._find_victim(set_index)
            
            # If victim is dirty, write it back to memory
            if self.cache[set_index][way][0] and self.cache[set_index][way][1]:
                self._writeback_line(set_index, way)
            
            # Allocate a new cache line
            self.cache[set_index][way] = [
                True,                   # valid
                True,                   # dirty
                tag,                    # tag
                1,                      # priority (initial)
                self.LONG_RRPV,         # RRPV (long interval)
                bytearray(self.line_size)  # empty data
            ]
        
        # Write the data
        if offset + size <= self.line_size:
            # Data fits in a single cache line
            self.cache[set_index][way][5][offset:offset+size] = data
            self.cache[set_index][way][1] = True  # Mark as dirty
        else:
            # Data spans multiple cache lines
            self.cache[set_index][way][5][offset:] = data[:self.line_size-offset]
            self.cache[set_index][way][1] = True  # Mark as dirty
            
            remaining_data = data[self.line_size-offset:]
            current_address = (line_address + 1) * self.line_size
            
            while remaining_data:
                bank, set_within_bank, tag, set_index = self._get_bank_set_way(current_address)
                
                # Check if bank is busy
                bank_busy, ready_cycle = self._is_bank_busy(bank, current_cycle)
                if bank_busy:
                    self.stats['bank_conflicts'] += 1
                
                # Mark the bank as busy for this operation
                self._set_bank_busy(bank, 1, max(current_cycle, ready_cycle))
                
                way = self._find_line(set_index, tag)
                
                if way is None:
                    # Allocate a new line
                    way = self._find_victim(set_index)
                    
                    # If victim is dirty, write it back to memory
                    if self.cache[set_index][way][0] and self.cache[set_index][way][1]:
                        self._writeback_line(set_index, way)
                    
                    # Allocate a new cache line
                    self.cache[set_index][way] = [
                        True,                   # valid
                        True,                   # dirty
                        tag,                    # tag
                        1,                      # priority (initial)
                        self.LONG_RRPV,         # RRPV (long interval)
                        bytearray(self.line_size)  # empty data
                    ]
                
                bytes_to_write = min(len(remaining_data), self.line_size)
                self.cache[set_index][way][5][:bytes_to_write] = remaining_data[:bytes_to_write]
                self.cache[set_index][way][1] = True  # Mark as dirty
                
                remaining_data = remaining_data[bytes_to_write:]
                current_address += self.line_size
        
        return True
    
    def consume(self, address, size, pe_id=None):
        """
        Read data from the cache and then invalidate it.
        This is used for partial output fibers that are consumed by the PE.
        
        Args:
            address: Starting address of the data to consume
            size: Size of the data in bytes
            pe_id: Processing Element ID performing the consumption
            
        Returns:
            Data read or None if not present
        """
        self.stats['consume_requests'] += 1
        current_cycle = self.stats['current_cycle']
        
        # First, read the data
        data = self.read(address, size, pe_id)
        
        if data is not None:
            # Then invalidate the lines
            line_address = address // self.line_size
            lines_to_invalidate = (size + (address % self.line_size) + self.line_size - 1) // self.line_size
            
            for i in range(lines_to_invalidate):
                current_address = (line_address + i) * self.line_size
                bank, set_within_bank, tag, set_index = self._get_bank_set_way(current_address)
                
                # Check if bank is busy
                bank_busy, ready_cycle = self._is_bank_busy(bank, current_cycle)
                if bank_busy:
                    self.stats['bank_conflicts'] += 1
                
                # Mark the bank as busy for this operation
                self._set_bank_busy(bank, 1, max(current_cycle, ready_cycle))
                
                way = self._find_line(set_index, tag)
                
                if way is not None:
                    # Clear the entry - no need to write back even if dirty since this is a consume operation
                    self.cache[set_index][way][0] = False  # Invalidate
                    self.cache[set_index][way][1] = False  # Clear dirty bit
        
        return data
    
    def flush(self):
        """
        Flush the entire cache, writing back dirty lines.
        
        Returns:
            Number of lines written back
        """
        writebacks = 0
        current_cycle = self.stats['current_cycle']
        
        for set_index in range(len(self.cache)):
            bank = set_index // self.sets_per_bank
            
            for way in range(self.associativity):
                if self.cache[set_index][way][0] and self.cache[set_index][way][1]:
                    # Dirty line - write back to memory
                    self._writeback_line(set_index, way)
                    writebacks += 1
                    
                    # Check if bank is busy
                    bank_busy, ready_cycle = self._is_bank_busy(bank, current_cycle)
                    if bank_busy:
                        self.stats['bank_conflicts'] += 1
                    
                    # Mark the bank as busy for this operation
                    self._set_bank_busy(bank, 1, max(current_cycle, ready_cycle))
                
                # Invalidate the line
                self.cache[set_index][way][0] = False
                self.cache[set_index][way][1] = False
        
        return writebacks
    
    def tick(self):
        """
        Advance FiberCache by one cycle.
        
        Returns:
            Status dictionary
        """
        self.stats['current_cycle'] += 1
        
        # Process any outstanding fetches that might have completed
        # In a real implementation, this would be handled by callbacks
        
        return {
            'cycle': self.stats['current_cycle'],
            'outstanding_fetches': len(self.outstanding_fetches)
        }
    
    def get_stats(self):
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        # Calculate hit rate
        total_requests = self.stats['read_requests'] + self.stats['write_requests'] + self.stats['consume_requests']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        stats = dict(self.stats)
        stats['hit_rate'] = hit_rate
        
        # Calculate utilization
        valid_lines = sum(1 for set_lines in self.cache for line in set_lines if line[0])
        total_lines = len(self.cache) * self.associativity
        stats['utilization'] = valid_lines / total_lines
        
        return stats