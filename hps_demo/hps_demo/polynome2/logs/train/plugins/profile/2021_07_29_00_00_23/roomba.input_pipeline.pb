	???? P@???? P@!???? P@	9(,?e:??9(,?e:??!9(,?e:??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???? P@??ݯ?@A?n/i??L@Y?ɐ??*	??"???c@2U
Iterator::Model::ParallelMapV2;s	????!??K?8@);s	????1??K?8@:Preprocessing2F
Iterator::Model???fd???!5Zlq?&H@)GXT??$??1N?????7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??]?????!?	6??V<@)R?d=???1?h??7@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??C????!DI?Nr!@)??C????1DI?Nr!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateg??e???!Gd1?0?0@))=?K?e??1H??J @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?B=}???!??&??@)?B=}???1??&??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipѐ?(????!˥??a?I@)5&?\R?}?1?η?3V@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?]??k??!N?s??2@)p?n???k?1N?4?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 10.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no98(,?e:??I?i??b?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??ݯ?@??ݯ?@!??ݯ?@      ??!       "      ??!       *      ??!       2	?n/i??L@?n/i??L@!?n/i??L@:      ??!       B      ??!       J	?ɐ???ɐ??!?ɐ??R      ??!       Z	?ɐ???ɐ??!?ɐ??b      ??!       JCPU_ONLYY8(,?e:??b q?i??b?X@