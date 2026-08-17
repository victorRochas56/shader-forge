IDEAS:

- proper freeing of resources,
also ability to save whole nodes (+ children, material, emitter etc) as a "prefab" that can be dragged and dropped into the scene.

- have resource loading on a seperate thread, 
allocate a tmp / stand-in resource handle that
gets swapped out with the real thing when it's loaded.

- test different compression levels

- look into making shadow maps less costly

- custom mesh format for instance info etc..

- improve scene serialization / deserialization, currently a bit spaghetti

- better tracking of memory usage on gpu, instances, LODs, texture memory etc..