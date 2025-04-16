class MemorySystem:
    """
    Memory system for Gamma accelerator.
    
    Models HBM-style memory with multiple channels and banks,
    tracking bandwidth utilization and enforcing bandwidth limits.
    """
    def __init__(self, bandwidth_GB_per_s=128, num_channels=16, banks_per_channel=16, 
                 line_size=64, latency_cycles=250, frequency_MHz=1000):
        """
        Initialize the memory system.
        
        Args:
            bandwidth_GB_per_s: Total memory bandwidth in GB/s
            num_channels: Number of HBM channels
            banks_per_channel: Number of banks per channel
            line_size: Cache line size in bytes
            latency_cycles: Base memory latency in cycles
            frequency_MHz: System frequency in MHz
        """
        self.total_bandwidth = bandwidth_GB_per_s
        self.num_channels = num_channels
        self.banks_per_channel = banks_per_channel
        self.total_banks = num_channels * banks_per_channel
        self.line_size = line_size
        self.base_latency = latency_cycles
        self.frequency = frequency_MHz
        
        # Calculate per-cycle bandwidth in bytes
        self.bytes_per_cycle = (bandwidth_GB_per_s * 1e9) / (frequency_MHz * 1e6)
        self.bytes_per_channel_per_cycle = self.bytes_per_cycle / num_channels
        
        # Bank state tracking
        self.bank_busy_until = [[0 for _ in range(banks_per_channel)] for _ in range(num_channels)]
        
        # Request queue per channel
        self.channel_queues = [[] for _ in range(num_channels)]
        
        # Bandwidth usage tracking
        self.bandwidth_used_per_cycle = []  # Historical bandwidth usage
        self.current_cycle = 0
        
        # Simple memory contents (for simulation)
        self.memory = {}  # addr -> data
        
        # Outstanding requests
        self.outstanding_requests = {}  # request_id -> (completion_cycle, callback)
        self.next_request_id = 0
        
        # Statistics
        self.stats = {
            'reads': 0,
            'writes': 0,
            'read_bytes': 0,
            'write_bytes': 0,
            'avg_bandwidth_utilization': 0,
            'max_bandwidth_utilization': 0,
            'bank_conflicts': 0,
            'avg_read_latency': 0,
            'total_read_latency': 0,
            'avg_queue_depth': 0,
            'total_queue_depth_samples': 0,
        }
    
    def _get_channel_and_bank(self, address):
        """
        Determine which channel and bank an address maps to.
        
        Args:
            address: Memory address
            
        Returns:
            Tuple of (channel, bank)
        """
        # Extract bits for channel and bank selection
        # This is a simplified mapping - real HBM would have more complex addressing
        line_address = address // self.line_size
        channel = line_address % self.num_channels
        bank = (line_address // self.num_channels) % self.banks_per_channel
        
        return channel, bank
    
    def _calculate_latency(self, channel, bank, is_read, size):
        """
        Calculate memory access latency based on bank state.
        
        Args:
            channel: Memory channel
            bank: Bank within the channel
            is_read: Whether this is a read operation
            size: Size of the transfer in bytes
            
        Returns:
            Latency in cycles
        """
        # Base latency plus any additional cycles due to bank being busy
        current_cycle = self.current_cycle
        bank_ready_cycle = self.bank_busy_until[channel][bank]
        
        # Check for bank conflict
        if bank_ready_cycle > current_cycle:
            self.stats['bank_conflicts'] += 1
        
        # Calculate when operation can start
        start_cycle = max(current_cycle, bank_ready_cycle)
        
        # Calculate transfer time
        bytes_per_cycle = self.bytes_per_channel_per_cycle
        transfer_cycles = (size + bytes_per_cycle - 1) // bytes_per_cycle
        
        # Total latency from now
        total_latency = (start_cycle - current_cycle) + self.base_latency + transfer_cycles
        
        # Update bank busy time
        self.bank_busy_until[channel][bank] = start_cycle + transfer_cycles
        
        return total_latency
    
    def _check_bandwidth_available(self, cycle, size):
        """
        Check if there's enough bandwidth available in the given cycle.
        
        Args:
            cycle: The cycle to check
            size: Size of the transfer in bytes
            
        Returns:
            Boolean indicating if bandwidth is available
        """
        # Extend bandwidth tracking array if needed
        while len(self.bandwidth_used_per_cycle) <= cycle:
            self.bandwidth_used_per_cycle.append(0)
        
        # Check if adding this transfer would exceed bandwidth
        if self.bandwidth_used_per_cycle[cycle] + size > self.bytes_per_cycle:
            return False
        
        return True
    
    def _allocate_bandwidth(self, start_cycle, size):
        """
        Allocate bandwidth for a transfer.
        
        Args:
            start_cycle: Starting cycle
            size: Size of the transfer in bytes
            
        Returns:
            Boolean indicating success
        """
        # Calculate how many cycles the transfer will take
        bytes_per_cycle = self.bytes_per_cycle
        bytes_remaining = size
        current_cycle = start_cycle
        
        while bytes_remaining > 0:
            # Extend bandwidth tracking array if needed
            while len(self.bandwidth_used_per_cycle) <= current_cycle:
                self.bandwidth_used_per_cycle.append(0)
            
            # Calculate how much we can transfer this cycle
            bytes_this_cycle = min(bytes_remaining, 
                                   bytes_per_cycle - self.bandwidth_used_per_cycle[current_cycle])
            
            # If we can't transfer anything this cycle, try the next one
            if bytes_this_cycle == 0:
                current_cycle += 1
                continue
            
            # Allocate bandwidth
            self.bandwidth_used_per_cycle[current_cycle] += bytes_this_cycle
            bytes_remaining -= bytes_this_cycle
            current_cycle += 1
        
        # Update statistics
        max_util = max(self.bandwidth_used_per_cycle) / self.bytes_per_cycle
        self.stats['max_bandwidth_utilization'] = max(self.stats['max_bandwidth_utilization'], max_util)
        
        return True
    
    def _get_next_request_id(self):
        """
        Generate a unique request ID.
        
        Returns:
            Unique request ID
        """
        request_id = self.next_request_id
        self.next_request_id += 1
        return request_id
    
    def read(self, address, size, callback=None):
        """
        Read data from memory.
        
        Args:
            address: Starting address
            size: Size in bytes
            callback: Function to call when read completes (optional)
            
        Returns:
            Request ID or data (if synchronous)
        """
        self.stats['reads'] += 1
        self.stats['read_bytes'] += size
        
        # Determine channel and bank
        channel, bank = self._get_channel_and_bank(address)
        
        # Calculate latency
        latency = self._calculate_latency(channel, bank, True, size)
        
        # Update read latency statistics
        self.stats['total_read_latency'] += latency
        self.stats['avg_read_latency'] = self.stats['total_read_latency'] / self.stats['reads']
        
        # Create a request object
        request = {
            'type': 'read',
            'address': address,
            'size': size,
            'channel': channel,
            'bank': bank,
            'start_cycle': self.current_cycle,
            'completion_cycle': self.current_cycle + latency,
            'callback': callback
        }
        
        # Add to channel queue
        self.channel_queues[channel].append(request)
        
        # Update queue depth statistics
        queue_depth = sum(len(queue) for queue in self.channel_queues)
        self.stats['total_queue_depth_samples'] += queue_depth
        self.stats['avg_queue_depth'] = self.stats['total_queue_depth_samples'] / (self.current_cycle + 1)
        
        # Allocate bandwidth
        self._allocate_bandwidth(self.current_cycle, size)
        
        # Generate request ID and register the outstanding request
        request_id = self._get_next_request_id()
        self.outstanding_requests[request_id] = (self.current_cycle + latency, callback)
        
        # Return request ID if callback provided, otherwise return data
        if callback:
            return request_id
        else:
            # For synchronous reads, create the data
            if address in self.memory:
                return self.memory[address:address+size]
            else:
                # Create some dummy data
                print("Data not in memory")
                return bytearray(size)
    
    def write(self, address, data, callback=None):
        """
        Write data to memory.
        
        Args:
            address: Starting address
            data: Data to write
            callback: Function to call when write completes (optional)
            
        Returns:
            Request ID or Boolean indicating success (if synchronous)
        """
        size = len(data)
        self.stats['writes'] += 1
        self.stats['write_bytes'] += size
        
        # Determine channel and bank
        channel, bank = self._get_channel_and_bank(address)
        
        # Calculate latency
        latency = self._calculate_latency(channel, bank, False, size)
        
        # Create a request object
        request = {
            'type': 'write',
            'address': address,
            'data': data,
            'size': size,
            'channel': channel,
            'bank': bank,
            'start_cycle': self.current_cycle,
            'completion_cycle': self.current_cycle + latency,
            'callback': callback
        }
        
        # Add to channel queue
        self.channel_queues[channel].append(request)
        
        # Update queue depth statistics
        queue_depth = sum(len(queue) for queue in self.channel_queues)
        self.stats['total_queue_depth_samples'] += queue_depth
        self.stats['avg_queue_depth'] = self.stats['total_queue_depth_samples'] / (self.current_cycle + 1)
        
        # Allocate bandwidth
        self._allocate_bandwidth(self.current_cycle, size)
        
        # Store the data
        self.memory[address] = data
        
        # Generate request ID and register the outstanding request
        request_id = self._get_next_request_id()
        self.outstanding_requests[request_id] = (self.current_cycle + latency, callback)
        
        # Return request ID if callback provided, otherwise return success
        if callback:
            return request_id
        else:
            return True
    
    def is_request_complete(self, request_id):
        """
        Check if a request has completed.
        
        Args:
            request_id: ID of the request
            
        Returns:
            Boolean indicating if the request is complete
        """
        if request_id not in self.outstanding_requests:
            return True
        
        completion_cycle, _ = self.outstanding_requests[request_id]
        return self.current_cycle >= completion_cycle
    
    def tick(self):
        """
        Advance memory system by one cycle.
        
        Returns:
            List of completed request IDs
        """
        self.current_cycle += 1
        completed_requests = []
        
        # Process completed requests
        for request_id, (completion_cycle, callback) in list(self.outstanding_requests.items()):
            if self.current_cycle >= completion_cycle:
                if callback:
                    callback()
                completed_requests.append(request_id)
                del self.outstanding_requests[request_id]
        
        # Process channel queues
        for channel in range(self.num_channels):
            # Remove completed requests from queue
            self.channel_queues[channel] = [req for req in self.channel_queues[channel] 
                                           if self.current_cycle < req['completion_cycle']]
        
        # Update bandwidth utilization statistics
        if self.current_cycle < len(self.bandwidth_used_per_cycle):
            total_util = sum(min(usage, self.bytes_per_cycle) for usage in self.bandwidth_used_per_cycle[:self.current_cycle+1])
            self.stats['avg_bandwidth_utilization'] = total_util / ((self.current_cycle + 1) * self.bytes_per_cycle)
        
        return completed_requests
    
    def get_stats(self):
        """
        Get memory system statistics.
        
        Returns:
            Dictionary of statistics
        """
        return self.stats


class MemoryController:
    """
    Memory controller that interfaces between Gamma components and the memory system.
    Handles request scheduling and bandwidth management.
    """
    def __init__(self, memory_system):
        """
        Initialize the memory controller.
        
        Args:
            memory_system: Reference to the memory system
        """
        self.memory = memory_system
        self.pending_requests = []  # Requests waiting for bandwidth
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'requests_queued': 0,
            'avg_queue_time': 0,
            'total_queue_time': 0
        }
    
    def read(self, address, size, callback=None, priority=0):
        """
        Read data from memory with optional callback.
        
        Args:
            address: Starting address
            size: Size in bytes
            callback: Function to call when read completes (optional)
            priority: Request priority (higher = more important)
            
        Returns:
            Request ID
        """
        self.stats['total_requests'] += 1
        
        # Create request record
        request = {
            'type': 'read',
            'address': address,
            'size': size,
            'callback': callback,
            'priority': priority,
            'queued_cycle': self.memory.current_cycle
        }
        
        # Try to submit directly to memory system
        request_id = self.memory.read(address, size, callback)
        
        return request_id
    
    def write(self, address, data, callback=None, priority=0):
        """
        Write data to memory with optional callback.
        
        Args:
            address: Starting address
            data: Data to write
            callback: Function to call when write completes (optional)
            priority: Request priority (higher = more important)
            
        Returns:
            Request ID
        """
        self.stats['total_requests'] += 1
        
        # Create request record
        request = {
            'type': 'write',
            'address': address,
            'data': data,
            'size': len(data),
            'callback': callback,
            'priority': priority,
            'queued_cycle': self.memory.current_cycle
        }
        
        # Try to submit directly to memory system
        request_id = self.memory.write(address, data, callback)
        
        return request_id
    
    def tick(self):
        """
        Advance memory controller by one cycle.
        
        Returns:
            Status dictionary
        """
        # First tick the memory system
        completed_requests = self.memory.tick()
        
        # Try to process any pending requests
        if self.pending_requests:
            # Sort by priority (higher priority first)
            self.pending_requests.sort(key=lambda r: -r['priority'])
            
            # Try to submit as many as possible
            remaining_requests = []
            for request in self.pending_requests:
                # Calculate queue time
                queue_time = self.memory.current_cycle - request['queued_cycle']
                self.stats['total_queue_time'] += queue_time
                
                # Submit based on request type
                if request['type'] == 'read':
                    request_id = self.memory.read(
                        request['address'], 
                        request['size'], 
                        request['callback']
                    )
                    if request_id is not None:
                        # Request accepted
                        self.stats['requests_queued'] += 1
                    else:
                        # Request still pending
                        remaining_requests.append(request)
                else:  # write
                    request_id = self.memory.write(
                        request['address'], 
                        request['data'], 
                        request['callback']
                    )
                    if request_id is not None:
                        # Request accepted
                        self.stats['requests_queued'] += 1
                    else:
                        # Request still pending
                        remaining_requests.append(request)
            
            # Update pending requests
            self.pending_requests = remaining_requests
        
        # Update statistics
        if self.stats['requests_queued'] > 0:
            self.stats['avg_queue_time'] = self.stats['total_queue_time'] / self.stats['requests_queued']
        
        return {
            'completed_requests': len(completed_requests),
            'pending_requests': len(self.pending_requests)
        }
    
    def get_stats(self):
        """
        Get memory controller statistics.
        
        Returns:
            Dictionary combining memory controller and memory system statistics
        """
        stats = dict(self.stats)
        stats.update(self.memory.get_stats())
        
        return stats