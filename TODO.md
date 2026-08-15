IDEAS:

- have resource loading on a seperate thread, 
allocate a tmp / stand-in resource handle that
gets swapped out with the real thing when it's loaded.

- test different compression levels

- look into making shadow maps less costly

- custom mesh format for instance info etc..

- improve scene serialization / deserialization, currently a bit spaghetti

- better tracking of memory usage on gpu, instances, LODs, texture memory etc..