import os, re
import numpy as np 

class DeepMedic(object):
    """
    Python wrapper object for DeepMedic: https://github.com/Kamnitsask/deepmedic/
    """
    def __init__(self,
        output_folder=".deepmedic/output/",
        n_classes=2, n_input_channels=1,
        n_fms_layer=[30,30,40,40,40,40,50,50], filter_sizes=[[3,3,3]],
        add_residual=[4,6,8], lower_rank=[], 
        subsample_path="copy", sub_fms_layer=[], sub_filter_sizes=[],
        subsample_factor=[3,3,3], sub_add_residual=[4,6,8], sub_lower_rank=[],
        fc_fms_layer=[150,150], fc1_filter_size=[3,3,3], fc_add_residual=[2],
        segment_train_size=[25,25,25], segment_val_size=[50,50,50], 
        segment_test_size=[100,100,100],
        batch_size=10, batch_size_val=50, batch_size_test=25, 
        dropout=0.02, sub_dropout=0.02, fc_dropout=[0.0,0.5,0.5],
        initializer="he_normal", activation="prelu", rollover_bn=60, 
        **kwargs):
        """
        Parameters:
        ----------
        subsample_path : ("copy", "custom", None)
            - if None, subsampled pathway is not used 
            - if "copy", subsampled pathway has same FMs/filters as normal
            - if "custom", must specify sub_fms_layer, sub_filter_sizes
        initializer : ("he_normal", "classic_normal") 
        activation : ("relu", "prelu")
        """
        self.output_folder = output_folder 
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder) 
        self.n_classes = n_classes 
        self.n_input_channels = n_input_channels
        self.n_fms_layer = n_fms_layer
        self.filter_sizes = filter_sizes
        if len(self.filter_sizes) == 1: 
            self.filter_sizes = filter_sizes * len(n_fms_layer) 
        self.add_residual = add_residual
        self.lower_rank = lower_rank
        self.subsample_path = subsample_path
        if self.subsample_path == "copy":
            self.sub_fms_layer = self.n_fms_layer 
            self.sub_filter_sizes = self.filter_sizes 
        else: 
            self.sub_fms_layer = sub_fms_layer 
            self.sub_filter_sizes = sub_filter_sizes
        if self.subsample_path == "custom": 
            if sub_fms_layer == []: 
                raise Exception("Specified custom subsampled pathway with empty feature maps: {}".format(sub_fms_layer))
            if sub_filter_sizes == []:
                raise Exception("Specified custom subsampled pathway with empty filters: {}".format(sub_fms_layer))
        self.subsample_factor = subsample_factor 
        self.sub_add_residual = sub_add_residual 
        self.sub_lower_rank = sub_lower_rank 
        self.fc_fms_layer = fc_fms_layer 
        self.fc1_filter_size = fc1_filter_size 
        self.fc_add_residual= fc_add_residual 
        self.segment_train_size = segment_train_size 
        self.segment_val_size = segment_val_size 
        self.segment_test_size = segment_test_size 
        self.batch_size = batch_size 
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        if type(dropout) != list: dropout = [dropout] 
        if len(dropout) == 1: 
            dropout = dropout * len(self.n_fms_layer) 
        self.dropout = dropout 
        if type(sub_dropout) != list: sub_dropout = [sub_dropout] 
        if len(sub_dropout) == 1: 
            sub_dropout = sub_dropout * len(self.sub_fms_layer) 
        self.sub_dropout = sub_dropout         
        if type(fc_dropout) != list: fc_dropout = [fc_dropout] 
        if len(fc_dropout) == 1: 
            fc_dropout = fc_dropout * (len(self.fc_fms_layer) + 1)
        self.fc_dropout = fc_dropout     
        self.initializer = initializer 
        self.activation = activation
        self.rollover_bn = rollover_bn

    def generate_model(self, model_name, gpu=True, 
        save_path=".deepmedic/configs/model/"):
        # Make sure deepMedicRun is in your $PATH
        if not os.path.exists(save_path): os.makedirs(save_path) 
        path_to_model_file = os.path.join(save_path,"modelConfig_{}.cfg".format(model_name))
        with open(path_to_model_file, "w") as f:
            f.write("modelName = '{}'\n".format(model_name))
            f.write("folderForOutput = '{}'\n".format(os.path.abspath(self.output_folder)))
            f.write("numberOfOutputClasses = {}\n".format(self.n_classes))
            f.write("numberOfInputChannels = {}\n".format(self.n_input_channels))
            f.write("numberFMsPerLayerNormal = {}\n".format(self.n_fms_layer))
            f.write("kernelDimPerLayerNormal = {}\n".format(self.filter_sizes))
            f.write("layersWithResidualConnNormal = {}\n".format(self.add_residual))
            f.write("lowerRankLayersNormal = {}\n".format(self.lower_rank))
            if self.subsample_path is not None: 
                subsample_boolean = True 
            f.write("useSubsampledPathway = {}\n".format(subsample_boolean))
            f.write("numberFMsPerLayerSubsampled = {}\n".format(self.sub_fms_layer))
            f.write("kernelDimPerLayerSubsampled = {}\n".format(self.sub_filter_sizes))
            f.write("subsampleFactor = {}\n".format(self.subsample_factor))
            f.write("layersWithResidualConnSubsampled = {}\n".format(self.sub_add_residual))
            f.write("lowerRankLayersSubsampled = {}\n".format(self.sub_lower_rank))
            f.write("numberFMsPerLayerFC = {}\n".format(self.fc_fms_layer))
            f.write("kernelDimFor1stFcLayer = {}\n".format(self.fc1_filter_size))
            f.write("layersWithResidualConnFC = {}\n".format(self.fc_add_residual))
            f.write("segmentsDimTrain = {}\n".format(self.segment_train_size))
            f.write("segmentsDimVal = {}\n".format(self.segment_val_size))
            f.write("segmentsDimInference = {}\n".format(self.segment_test_size))
            f.write("batchSizeTrain = {}\n".format(self.batch_size))
            f.write("batchSizeVal = {}\n".format(self.batch_size_val))
            f.write("batchSizeInfer = {}\n".format(self.batch_size_test))
            f.write("dropoutRatesNormal = {}\n".format(self.dropout))
            f.write("dropoutRatesSubsampled = {}\n".format(self.sub_dropout))
            f.write("dropoutRatesFc = {}\n".format(self.fc_dropout))
            if self.initializer == "he_normal":
                initial_int = 1
            elif self.initializer == "classic_normal":
                initial_int = 0 
            f.write("initializeClass0OrDelving1 = {}\n".format(initial_int))
            if self.activation == "prelu": 
                activate_int = 1 
            elif self.activation == "relu": 
                activate_int = 0 
            f.write("relu0orPrelu1 = {}\n".format(activate_int)) 
            f.write("rollAverageForBNOverThatManyBatches = {}".format(self.rollover_bn))
        self.model_config_file = path_to_model_file 
        if gpu: 
            dev = "-dev gpu"
        else: 
            dev = ""
        os.system("deepMedicRun {} -newModel {}".format(dev, self.model_config_file))
        with open(os.path.join(self.output_folder, "logs/{}.txt".format(model_name)), "r") as f: 
            x = f.readlines() 
            x = [_.strip() for _ in x] 
            model_save_line = [_ for _ in x if bool(re.search("Saving the model", _))]
            which_lines = np.argwhere(np.asarray(x) == model_save_line[0])
            saved_model = x[which_lines[-1][0] + 1].split("Saving network to: ")[1] 
        self.saved_model = saved_model 
    
    def set_model_to_train(self, model_to_train):
        self.saved_model = model_to_train 

    def set_training_parameters(self, X, y, masks, 
        save_path=".deepmedic/output/train/",
        default_sampling=True, sampling_type=0, class_sampling_wt=[0.5,0.5],
        sampling_wt_maps=None, epochs=35, subepochs=20, 
        num_cases_per_subepoch=50, 
        num_segments_per_subepoch=1000, 
        base_lr=1e-3, optimizer="adam", momentum=0.9, momentum_type="nesterov", 
        norm_momentum=True, rho_rms=0.9, eps_rms=1e-4, eps_adam=1e-8, 
        lr_schedule_type=2, scale_lr_factor=2.0, 
        lr_schedule=[12,16,19,22,25,28,31,34,37,40,43,46], 
        reflect_images=[True,False,False], 
        int_augment=False, loc_int_augm=[0,0.1], scale_int_augm=[1.,0.05],
        l1=0.0, l2=1e-4, freeze=[], sub_freeze=[], fc_freeze=[]):
        self.model_output_folder = save_path 
        if not os.path.exists(self.model_output_folder): 
            os.makedirs(self.model_output_folder)
        self.X_train = X 
        self.y_train = y 
        self.masks_train = masks 
        self.default_sampling = default_sampling
        self.sampling_type = sampling_type
        self.class_sampling_wt = class_sampling_wt
        self.sampling_wt_maps = sampling_wt_maps
        self.epochs = epochs 
        self.subepochs = subepochs 
        self.num_cases_per_subepoch = num_cases_per_subepoch
        self.num_segments_per_subepoch = num_segments_per_subepoch
        self.base_lr = base_lr 
        self.optimizer = optimizer
        self.momentum = momentum
        self.momentum_type = momentum_type 
        self.norm_momentum = norm_momentum 
        self.rho_rms = rho_rms 
        self.eps_rms = eps_rms 
        self.eps_adam = eps_adam 
        self.lr_schedule_type = lr_schedule_type
        self.scale_lr_factor = scale_lr_factor
        self.lr_schedule = lr_schedule 
        self.reflect_images = reflect_images
        self.int_augment = int_augment 
        self.loc_int_augm = loc_int_augm
        self.scale_int_augm = scale_int_augm
        self.l1 = l1 
        self.l2 = l2 
        self.freeze = freeze 
        self.sub_freeze = sub_freeze 
        self.fc_freeze = fc_freeze 

    def set_validation_parameters(self, X, y, masks, names, 
        val_during_train=True, 
        full_inference=True, 
        num_segments_per_subepoch=5000, 
        default_sampling=True, 
        sampling_type=1, 
        class_sampling_wt=[0.5,0.5],
        sampling_wt_maps=None, 
        val_btwn_epochs=2,
        save_segmentations=False, 
        save_prob_maps=[False, True], 
        pad_input=True):
        self.X_val = X 
        self.y_val = y 
        self.masks_val = masks 
        self.val_names = names 
        self.val_during_train = True 
        self.full_inference = True 
        self.val_num_segments_per_subepoch = num_segments_per_subepoch
        self.val_default_sampling = default_sampling
        self.val_sampling_type = sampling_type 
        self.val_class_sampling_wt = class_sampling_wt 
        self.val_sampling_wt_maps = sampling_wt_maps
        self.val_btwn_epochs = val_btwn_epochs
        self.save_val_segmentations = save_segmentations 
        self.save_val_prob_maps = save_prob_maps 
        self.pad_val_input = pad_input 

    def train(self, session_name, gpu=True, save_path=".deepmedic/configs/train/"): 
        if not os.path.exists(save_path): os.makedirs(save_path) 
        path_to_train_file = os.path.join(save_path,"trainConfig_{}.cfg".format(session_name))
        with open(path_to_train_file, "w") as f:
            f.write("sessionName = '{}'\n".format(session_name))
            f.write("folderForOutput = '{}'\n".format(os.path.abspath(self.output_folder)))
            f.write("cnnModelFilePath = '{}'\n".format(os.path.abspath(self.model_output_folder)))
            f.write("channelsTraining = {}\n".format(self.X_train))
            f.write("gtLabelsTraining = '{}'\n".format(self.y_train))
            f.write("roiMasksTraining = '{}'\n".format(self.masks_train))
            f.write("useDefaultTrainingSamplingFromGtAndRoi = {}\n".format(self.default_sampling))
            f.write("typeOfSamplingForTraining = {}\n".format(self.sampling_type))
            f.write("proportionOfSamplesToExtractPerCategoryTraining = {}\n".format(self.class_sampling_wt))
            f.write("weightedMapsForSamplingEachCategoryTrain = {}\n".format(self.sampling_wt_maps))
            f.write("numberOfEpochs = {}\n".format(self.epochs))
            f.write("numberOfSubepochs = {}\n".format(self.subepochs))
            f.write("numOfCasesLoadedPerSubepoch = {}\n".format(self.num_cases_per_subepoch))
            f.write("numberTrainingSegmentsLoadedOnGpuPerSubep = {}\n".format(self.num_segments_per_subepoch))
            f.write("stable0orAuto1orPredefined2orExponential3LrSchedule = {}\n".format(self.lr_schedule_type))
            f.write("whenDecreasingDivideLrBy = {}\n".format(self.scale_lr_factor))
            f.write("predefinedSchedule = {}\n".format(self.lr_schedule))
            f.write("reflectImagesPerAxis = {}\n".format(self.reflect_images))
            f.write("performIntAugm = {}\n".format(self.int_augment))
            f.write("sampleIntAugmShiftWithMuAndStd = {}\n".format(self.loc_int_augm))
            f.write("sampleIntAugmMultiWithMuAndStd = {}\n".format(self.scale_int_augm))
            f.write("learningRate = {}\n".format(self.base_lr))
            if self.optimizer == "sgd": 
                opt_int = 0 
            elif self.optimizer == "adam": 
                opt_int = 1 
            elif self.optimizer == "rmsprop": 
                opt_int = 2
            f.write("sgd0orAdam1orRms2 = {}\n".format(opt_int))
            if self.momentum_type == "classic": 
                mom_int = 0 
            elif self.momentum_type == "nesterov": 
                mom_int = 1
            f.write("classicMom0OrNesterov1 = {}\n".format(mom_int))
            f.write("momentumValue = {}\n".format(self.momentum))
            if self.norm_momentum: 
                norm_mom_int = 1 
            else: 
                norm_mom_int = 0
            f.write("momNonNorm0orNormalized1 = {}\n".format(norm_mom_int))
            f.write("rhoRms = {}\n".format(self.rho_rms))
            f.write("epsilonRms = {}\n".format(self.eps_rms))
            f.write("epsilonAdam = {}\n".format(self.eps_adam))
            f.write("L1_reg = {}\n".format(self.l1))
            f.write("L2_reg = {}\n".format(self.l2))
            f.write("layersToFreezeNormal = {}\n".format(self.freeze))
            f.write("layersToFreezeSubsampled = {}\n".format(self.sub_freeze))
            f.write("layersToFreezeFC = {}\n".format(self.fc_freeze))
            f.write("performValidationOnSamplesThroughoutTraining = {}\n".format(self.val_during_train))
            f.write("performFullInferenceOnValidationImagesEveryFewEpochs = {}\n".format(self.full_inference))
            f.write("channelsValidation = {}\n".format(self.X_val))
            f.write("gtLabelsValidation = '{}'\n".format(self.y_val))
            f.write("numberValidationSegmentsLoadedOnGpuPerSubep = {}\n".format(self.val_num_segments_per_subepoch))
            f.write("roiMasksValidation = '{}'\n".format(self.masks_val))
            f.write("useDefaultUniformValidationSampling = {}\n".format(self.val_default_sampling))
            f.write("typeOfSamplingForVal = {}\n".format(self.val_sampling_type))
            f.write("proportionOfSamplesToExtractPerCategoryVal = {}\n".format(self.val_class_sampling_wt))
            f.write("weightedMapsForSamplingEachCategoryVal = {}\n".format(self.val_sampling_wt_maps))
            f.write("numberOfEpochsBetweenFullInferenceOnValImages = {}\n".format(self.val_btwn_epochs))
            f.write("saveSegmentationVal = {}\n".format(self.save_val_segmentations))
            f.write("saveProbMapsForEachClassVal = {}\n".format(self.save_val_prob_maps))
            f.write("namesForPredictionsPerCaseVal = '{}'\n".format(self.val_names)) 
            f.write("padInputImagesBool = {}\n".format(self.pad_val_input))
        self.train_file = path_to_train_file 
        if gpu: 
            dev = "-dev gpu"
        else: 
            dev = ""
        os.system("deepMedicRun {} -train {} -model {}".format(dev, self.train_file, self.saved_model))

    def set_model_to_test(self, model_to_test):
        # Make sure you specify the FULL path
        self.test_model = model_to_test

    def set_testing_parameters(self, X, y, masks, names,
        save_path="./deepmedic/output/test/",
        save_segmentations=False, save_prob_maps=[False, True], 
        save_individual_fms=False, save_all_fms_together=False, 
        min_max_indices_fms=[], min_max_indices_fms_sub=[], 
        min_max_indices_fms_fc=[], pad_input=True):
        """
        y : str
            - specify None if no test labels exist
        """
        if not os.path.exists(save_path): os.makedirs(save_path)
        self.X_test = X 
        self.y_test = y 
        self.masks_test = masks 
        self.test_names = names
        self.test_output_folder = save_path
        self.save_test_segmentations = save_segmentations
        self.save_test_prob_maps = save_prob_maps
        self.save_test_individual_fms = save_individual_fms 
        self.save_test_all_fms = save_all_fms_together
        self.test_min_max_indices_fms = min_max_indices_fms
        self.test_min_max_indices_fms_sub = min_max_indices_fms_sub
        self.test_min_max_indices_fms_fc = min_max_indices_fms_fc
        self.pad_test_input = pad_input 

    def test(self, session_name, gpu=True, save_path=".deepmedic/configs/test/"):
        if not os.path.exists(save_path): os.makedirs(save_path)
        path_to_test_file = os.path.join(save_path,"testConfig_{}.cfg".format(session_name))
	with open(path_to_test_file, "w") as f:
            f.write("sessionName = '{}'\n".format(session_name))
            f.write("folderForOutput = '{}'\n".format(os.path.abspath(self.test_output_folder)))
            f.write("cnnModelFilePath = '{}'\n".format(self.test_model))
            f.write("channels = {}\n".format(self.X_test))
            f.write("namesForPredictionsPerCase = '{}'\n".format(self.test_names))
            f.write("roiMasks = '{}'\n".format(self.masks_test))
            if self.y_test is not None: 
                f.write("gtLabels = '{}'\n".format(self.y_test))
            f.write("saveSegmentation = {}\n".format(self.save_test_segmentations))
            f.write("saveProbMapsForEachClass = {}\n".format(self.save_test_prob_maps))
            f.write("saveIndividualFms = {}\n".format(self.save_test_individual_fms))
            f.write("saveAllFmsIn4DimImage = {}\n".format(self.save_test_all_fms))
            f.write("minMaxIndicesOfFmsToSaveFromEachLayerOfNormalPathway = {}\n".format(self.test_min_max_indices_fms))
            f.write("minMaxIndicesOfFmsToSaveFromEachLayerOfSubsampledPathway = {}\n".format(self.test_min_max_indices_fms_sub))
            f.write("minMaxINdicesOfFmsToSaveFromEachLayerOfFullyConnectedPathway = {}\n".format(self.test_min_max_indices_fms_fc))
            f.write("padInput = {}\n".format(self.pad_test_input))
        self.test_file = path_to_test_file 
        if gpu: 
            dev = "-dev gpu"
        else:
            dev = ""
        os.system("deepMedicRun {} -test {}".format(dev, self.test_file))

