	???n?d@???n?d@!???n?d@	???n?l?????n?l??!???n?l??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???n?d@e?<$??AS?
c?d@Yap??/??*	w?????@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateAgҦ??(@!g!+i??X@)V?6???(@1B?? ??X@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?q??>s??!sq??I??)????????1@??Vs???:Preprocessing2U
Iterator::Model::ParallelMapV2W??x????!&???????)W??x????1&???????:Preprocessing2F
Iterator::Model??Dg?E??!"??O/??)/j?? ߝ?1=?D???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice=?!7???!CHH?????)=?!7???1CHH?????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?;?y?)@!?? a??X@)???s????1??????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??"??J??!???V,??)??"??J??1???V,??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?]gC??(@!?????X@)O崧??x?1?J??-{??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???n?l??I?M????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	e?<$??e?<$??!e?<$??      ??!       "      ??!       *      ??!       2	S?
c?d@S?
c?d@!S?
c?d@:      ??!       B      ??!       J	ap??/??ap??/??!ap??/??R      ??!       Z	ap??/??ap??/??!ap??/??b      ??!       JCPU_ONLYY???n?l??b q?M????X@