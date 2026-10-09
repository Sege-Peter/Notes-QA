# Rate Limiting Strategies

## Token Bucket
Token-bucket rate limiting is preferable to fixed windows because it smooths bursty traffic without rejecting legitimate spikes. Tokens refill at a constant rate up to a bucket capacity.

## Fixed Window Counter
Redis INCR + EXPIRE pattern is a common way to implement a simple fixed-window limiter, but warned it can allow up to 2x the intended rate at window boundaries.

## Leaky Bucket
Leaky bucket processes requests at a constant output rate, smoothing out bursts by queueing incoming traffic.
