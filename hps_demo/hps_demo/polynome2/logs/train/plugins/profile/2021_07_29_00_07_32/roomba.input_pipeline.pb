	?*Q?Do@?*Q?Do@!?*Q?Do@	??????????????!???????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?*Q?Do@L???^@A$?`S??_@Y?q?Pi??*	??C?l?s@2U
Iterator::Model::ParallelMapV2????????!K???1?7@)????????1K???1?7@:Preprocessing2F
Iterator::Model?z??9y??!??? 2?E@)?f????13??%2?3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?xwd?6??!?{4l?<@)bi?G5???1R????3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatt]???Ա?!p:??"6@)???GS??1M(??
32@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice?G??!??Pi? "@)?G??1??Pi? "@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip)x
?R???!?E??OL@)????G??1+M??5O@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_??x?Z??!?p?Gx@)_??x?Z??1?p?Gx@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???s????!?=m?i>@)ѭ????t?1Q"??Y???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 49.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???????I?IzR?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	L???^@L???^@!L???^@      ??!       "      ??!       *      ??!       2	$?`S??_@$?`S??_@!$?`S??_@:      ??!       B      ??!       J	?q?Pi???q?Pi??!?q?Pi??R      ??!       Z	?q?Pi???q?Pi??!?q?Pi??b      ??!       JCPU_ONLYY???????b q?IzR?X@