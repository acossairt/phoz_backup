	?lV}??@@?lV}??@@!?lV}??@@	(?: ???(?: ???!(?: ???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?lV}??@@?{?_????A?k?ˆ@@Y?X?????*U-???h@)       =2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???=?>??!????}?;@)'?E'K???19sBA7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate
.V?`??!GtC?ɐ=@)?;??????1?B??&?4@:Preprocessing2U
Iterator::Model::ParallelMapV2???ꫫ??!???UV2@)???ꫫ??1???UV2@:Preprocessing2F
Iterator::Model?Քd??!a?W͘?A@)?'?>???1?f???>1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice ???Qc??!?bb{F"@) ???Qc??1?bb{F"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip:?Fv?e??!O!T??P@)\?	??b??1_?? @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?Y,E??!O?N???@)?Y,E??1O?N???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??S:X???!???m?@)???_vOn?1???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9(?: ???I???f?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?{?_?????{?_????!?{?_????      ??!       "      ??!       *      ??!       2	?k?ˆ@@?k?ˆ@@!?k?ˆ@@:      ??!       B      ??!       J	?X??????X?????!?X?????R      ??!       Z	?X??????X?????!?X?????b      ??!       JCPU_ONLYY(?: ???b q???f?X@