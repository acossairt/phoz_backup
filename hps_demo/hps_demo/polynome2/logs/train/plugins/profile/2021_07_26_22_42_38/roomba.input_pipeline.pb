	????^?@????^?@!????^?@	??????????????!???????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$????^?@?4?;?@A6#?ܥ??@Y??<I??7@*	? ?r?i?@2U
Iterator::Model::ParallelMapV2?-;???&@!?*3?g?H@)?-;???&@1?*3?g?H@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate)????@!?{???@@)E|f@1.?&?I?@@:Preprocessing2F
Iterator::Model????=.@!`?!?_P@)?v?@1}? ?:]0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice=b??BW??!4W??'??)=b??BW??14W??'??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?Ü?M??!??+?????)???-???1??m1ݕ??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipޑ????@!A_???AA@)??։????1???XgY??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;ŪA?ۍ?!???B??);ŪA?ۍ?1???B??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?*??@!?AT?@@)*??g\8??1$??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???????I?C????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?4?;?@?4?;?@!?4?;?@      ??!       "      ??!       *      ??!       2	6#?ܥ??@6#?ܥ??@!6#?ܥ??@:      ??!       B      ??!       J	??<I??7@??<I??7@!??<I??7@R      ??!       Z	??<I??7@??<I??7@!??<I??7@b      ??!       JCPU_ONLYY???????b q?C????X@