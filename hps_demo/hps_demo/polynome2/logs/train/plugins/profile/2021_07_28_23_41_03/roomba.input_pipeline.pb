	?Xz??@?Xz??@!?Xz??@	??>???@??>???@!??>???@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?Xz??@??G?3?@A?5\????@Y??!?:<@*	?&1?E?@2F
Iterator::Model4M?~2?5@!A??ۈS@)H?????2@1?7??P@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate????c@!!?"?\?5@)S?
c?@1??+@?n5@:Preprocessing2U
Iterator::Model::ParallelMapV2[????@!xc?=%@)[????@1xc?=%@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??q?߅??!?v???m??)؁sF????1?S?????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice?Nyt#,??!?ϔ{?D??)?Nyt#,??1?ϔ{?D??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??&??k@!???I??5@)???[??1?????ͳ?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorf???-=??!I?U}??)f???-=??1I?U}??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapk???u@!?]z??5@)S?Q?Gt?1qw??'??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??>???@I?ܕ?W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??G?3?@??G?3?@!??G?3?@      ??!       "      ??!       *      ??!       2	?5\????@?5\????@!?5\????@:      ??!       B      ??!       J	??!?:<@??!?:<@!??!?:<@R      ??!       Z	??!?:<@??!?:<@!??!?:<@b      ??!       JCPU_ONLYY??>???@b q?ܕ?W@