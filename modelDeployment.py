import detectCroppedSeg3DKerasDR_predict_ha
import streamlit as st

@st.cache
def runDeepSegmentationModel(organTarget, img):

    # model parameters
    params = {};
    params['TestSetNum'] = 1;
    params['tpUsed'] = 50;
    params['tDim'] = params['tpUsed'];
    params['PcUsed'] = 1;
    # params['visualizeResults']= 0;
    # params['visSlider']= 0;
    params['deepReduction'] = 0;

    params['networkToUseDetect'] = 'rbUnet'  # 'denseNet'; #'tNet'; #'Unet' #meshNet
    params['networkToUseSegment'] = 'tNet'  # 'denseNet'; #'rbUnet' # 'Unet' #meshNet

    if params['PcUsed'] == 1:
        tDim = 5;
        params['tDim'] = tDim;

    #pName = imagesAddress
    baseline = '1';
    
    if organTarget == 'Liver':
        params['selectedEpochDetect'] = '30000';
        params['selectedEpochSegment'] = '31735';

        # call the model to detect and segment and return the mask
        maskDetect, boxDetect, kidneyNone, vol4D0, vol4Dpcs, zDimOri = detectCroppedSeg3DKerasDR_predict_ha.singlePatientDetection(img, int(baseline), params, 'Liver');
        maskSegment, plotMask = detectCroppedSeg3DKerasDR_predict_ha.singlePatientSegmentation(params, img, maskDetect, boxDetect, kidneyNone, vol4D0, vol4Dpcs, zDimOri, 'Liver');

    if organTarget == 'Psoas':
        reconMethod = 'SCAN';
        params['selectedEpochDetect'] = '32755';
        params['selectedEpochSegment'] = '96000';
        vol4D00, oriKM, boxDetect0, _, _ = funcs_ha_use.readData4(img, reconMethod, 1, 'Psoas');
        
        maskDetect0 = oriKM.copy()
        kidneyNone0 = np.nonzero(np.sum(boxDetect0, axis=1) == 0);  # right/left
        if kidneyNone0[0].size != 0:
            kidneyNone0 = np.nonzero(np.sum(boxDetect0, axis=1) == 0)[0][0];  # right/left

        # call the model to detect and segment and return the mask
        maskDetect, boxDetect, kidneyNone, vol4D0, vol4Dpcs, zDimOri = detectCroppedSeg3DKerasDR_predict_ha.singlePatientDetection(img, int(baseline), params, 'Psoas');
        maskSegment, plotMask = detectCroppedSeg3DKerasDR_predict_ha.singlePatientSegmentation(params, img, maskDetect0, boxDetect0, kidneyNone0, vol4D0, vol4Dpcs, zDimOri, 'Psoas');

        
        return maskSegment, plotMask
