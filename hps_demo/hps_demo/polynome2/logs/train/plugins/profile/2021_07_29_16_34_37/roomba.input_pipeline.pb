	?B:<D??@?B:<D??@!?B:<D??@	_\{J?
??_\{J?
??!_\{J?
??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?B:<D??@B$C??g??AW]?j???@Y]??m5??*	??~j<??@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?	?/7@!?Rju??X@)??jdW&7@1???Q?X@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???{????!?H?
??)ywd?6???1??y?|???:Preprocessing2U
Iterator::Model::ParallelMapV2??O?ޤ?!:Ve2?W??)??O?ޤ?1:Ve2?W??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice?t><K???!Iݍ?i???)?t><K???1Iݍ?i???:Preprocessing2F
Iterator::ModelQ????+??!O??f???)?????x??1d??o???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipA}˜.G7@!?K??y?X@)?ꫫ???1??պ????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??5>????!??!?f??)??5>????1??!?f??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapX???T07@!?????X@)_???:Ts?1??9???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9_\{J?
??IJX?V_?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	B$C??g??B$C??g??!B$C??g??      ??!       "      ??!       *      ??!       2	W]?j???@W]?j???@!W]?j???@:      ??!       B      ??!       J	]??m5??]??m5??!]??m5??R      ??!       Z	]??m5??]??m5??!]??m5??b      ??!       JCPU_ONLYY_\{J?
??b qJX?V_?X@