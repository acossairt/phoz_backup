	?]?VdŊ@?]?VdŊ@!?]?VdŊ@	oE*H7e??oE*H7e??!oE*H7e??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?]?VdŊ@I?p???A?u?+???@Y?Q??????*	?Z?ב@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate׾?^?s??!M??D;?T@)???l????1,????T@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?*?]gC??!όEA?@)?ó??1c;v??@:Preprocessing2U
Iterator::Model::ParallelMapV2h$B#ظ??!?nh??@)h$B#ظ??1?nh??@:Preprocessing2F
Iterator::Model'??d?V??!???ؑ2@)?熦????1o%?CN@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice3P?>??!D??WU
@)3P?>??1D??WU
@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip}[?T???!??s??,W@)??;Mf???1?h?T?W@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?G????!?E=?????)?G????1?E=?????:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap2;?ީ??!????F?T@)??9]{?1?L`????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9nE*H7e??I\}????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	I?p???I?p???!I?p???      ??!       "      ??!       *      ??!       2	?u?+???@?u?+???@!?u?+???@:      ??!       B      ??!       J	?Q???????Q??????!?Q??????R      ??!       Z	?Q???????Q??????!?Q??????b      ??!       JCPU_ONLYYnE*H7e??b q\}????X@