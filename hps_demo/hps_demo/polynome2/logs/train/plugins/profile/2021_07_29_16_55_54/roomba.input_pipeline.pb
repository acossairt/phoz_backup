	?G?z?u@?G?z?u@!?G?z?u@	Ruf?@Ruf?@!Ruf?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?G?z?u@?ZӼ???A?)?J}t@Y??Dׅ_6@*	??/ѻ?@2U
Iterator::Model::ParallelMapV2?SW>??3@!?$~?kG@)?SW>??3@1?$~?kG@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??쟧4@!??P䡡G@)?Nw?x?1@1?ąh?D@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice.X?x?@!??`??9@).X?x?@1??`??9@:Preprocessing2F
Iterator::ModelP?<?N6@!??n§CJ@)b?*?3R@1!9V"2?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?_u?Hg??!?w???P??)?NGɫ??1<7??J??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipP??W(4@!&?=X?G@)q?Qew??1?O$????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?đ"???!?z??0??)?đ"???1?z??0??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???	4@!?o5?L?G@)؞Y??v?1e?H?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9Ruf?@I???>`W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?ZӼ????ZӼ???!?ZӼ???      ??!       "      ??!       *      ??!       2	?)?J}t@?)?J}t@!?)?J}t@:      ??!       B      ??!       J	??Dׅ_6@??Dׅ_6@!??Dׅ_6@R      ??!       Z	??Dׅ_6@??Dׅ_6@!??Dׅ_6@b      ??!       JCPU_ONLYYRuf?@b q???>`W@