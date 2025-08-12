import * as tf from '@tensorflow/tfjs';
import * as mobilenet from 'tensorflow-models/mobilenet';
import * as qna from '@tensorflow-models/qna';

export const SEQ_LENGTH = 10;

export const RNNModel = {
    name: 'RNNModel',
    create: (charsetSize: number): tf.LayersModel => {
        const model = tf.sequential();
        model.add(tf.layers.embedding({ inputDim: charsetSize, outputDim: 16, inputLength: SEQ_LENGTH }));
        model.add(tf.layers.simpleRNN({ units: 128 }));
        model.add(tf.layers.dense({ units: charsetSize, activation: 'softmax' }));
        model.compile({
            optimizer: tf.train.adam(),
            loss: 'categoricalCrossentropy',
        });
        return model;
    }
};

export const LSTMModel = {
    name: 'LSTMModel',
    create: (charsetSize: number): tf.LayersModel => {
        const model = tf.sequential();
        model.add(tf.layers.embedding({ inputDim: charsetSize, outputDim: 16, inputLength: SEQ_LENGTH }));
        model.add(tf.layers.lstm({ units: 128 }));
        model.add(tf.layers.dense({ units: charsetSize, activation: 'softmax' }));
        model.compile({
            optimizer: tf.train.adam(),
            loss: 'categoricalCrossentropy',
        });
        return model;
    }
}

export const MobileNetV3Model = {
    name: 'MobileNetV3Model',
    create: async (inputResolution: number = 224, modelArchitecture: string = 'small-075') => {
        const url = `https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small-075_224/classification/5/default/1`;
        const model = await tf.loadGraphModel(url, { fromTFHub: true });
        return model;
    }
}

export const MobileNetV3Model2 = {
    name: 'MobileNetV3Model2',
    create: async () => {
        const model = await mobilenet.load();
        return model;
    }
}

export const CocoSsdModel = {
    name: 'CocoSsdModel',
    create: async () => {
        const url = `https://storage.googleapis.com/tfjs-models/savedmodel/ssd_mobilenet_v1/model.json`
        const model = await tf.loadGraphModel(url);
        return model;
    }
}

export const QnaModel = {
    name: 'QnaModel',
    create: async () => {
        const model = await qna.load();
        return model;
    }
}

export const ModelRegistry = {
    rnn: RNNModel,
    lstm: LSTMModel,
    mobilenet: MobileNetV3Model2,
    cocoSsd: CocoSsdModel,
    qna: QnaModel,
};
