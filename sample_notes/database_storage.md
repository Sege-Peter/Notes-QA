# Database Storage Engines

## Write-Ahead Logging
A Write-Ahead Log (WAL) provides durability in database systems. Changes are appended to the log on disk before modifying memory pages, allowing crash recovery via replay.

## LSM Trees
Log-Structured Merge-trees buffer writes in an in-memory memtable and flush sorted runs to SSTables on disk. Compaction merges runs in the background.

## B-Trees
B-Trees organize data into fixed-size blocks (pages), offering balanced O(log n) lookups and updates, ideal for block-based storage.
