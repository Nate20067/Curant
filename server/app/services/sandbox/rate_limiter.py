#Rate limiter for web operations in sandbox
#Provides thread-safe rate limiting per domain to prevent API abuse
import time
import logging
from collections import defaultdict
from threading import Lock


class RateLimiter:
    """Thread-safe rate limiter for web operations"""
    
    def __init__(self, max_calls: int = 5, time_window: float = 60.0):
        #Initialize rate limiter
        #max_calls: Maximum number of calls allowed within time_window
        #time_window: Time window in seconds
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = defaultdict(list)  #domain -> list of timestamps
        self.lock = Lock()
    
    def _clean_old_calls(self, domain: str, current_time: float):
        #Remove timestamps older than time_window
        cutoff = current_time - self.time_window
        self.calls[domain] = [t for t in self.calls[domain] if t > cutoff]
    
    def is_allowed(self, domain: str) -> tuple:
        #Check if a call is allowed for the given domain
        #Returns (is_allowed, wait_time) - wait_time is 0 if allowed, otherwise seconds to wait
        with self.lock:
            current_time = time.time()
            self._clean_old_calls(domain, current_time)
            
            if len(self.calls[domain]) < self.max_calls:
                return True, 0.0
            
            #Calculate wait time until oldest call expires
            oldest_call = min(self.calls[domain])
            wait_time = (oldest_call + self.time_window) - current_time
            return False, max(0, wait_time)
    
    def record_call(self, domain: str):
        #Record a call for the given domain
        with self.lock:
            current_time = time.time()
            self._clean_old_calls(domain, current_time)
            self.calls[domain].append(current_time)
    
    def wait_if_needed(self, domain: str, max_wait: float = 30.0):
        #Wait if rate limit is exceeded, up to max_wait seconds
        #domain: Domain to check rate limit for
        #max_wait: Maximum time to wait before raising exception
        #Raises RuntimeError if wait time exceeds max_wait
        is_allowed, wait_time = self.is_allowed(domain)
        
        if not is_allowed:
            if wait_time > max_wait:
                raise RuntimeError(
                    f"Rate limit exceeded for {domain}. "
                    f"Would need to wait {wait_time:.1f}s (max: {max_wait}s)"
                )
            
            logging.info(f"Rate limit hit for {domain}, waiting {wait_time:.1f}s")
            time.sleep(wait_time)