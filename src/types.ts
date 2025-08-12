import * as tf from '@tensorflow/tfjs'
import {TensorContainer} from "@tensorflow/tfjs";

export type TfBackend = 'webgpu' | 'webgl' | 'cpu' | 'wasm';
export type TrainingText = 'hamlet';
export type ModelType = 'rnn' | 'nlp' | 'mobileNetV3' | 'cocossd' | 'matmulti';

export type ProfileInfo = {
    newBytes: number; newTensors: number; peakBytes: number;
    kernels: KernelInfo[];
    result: TensorContainer;
    kernelNames: string[];
};

type KernelInfo = {
    name: string; bytesAdded: number; totalBytesSnapshot: number;
    tensorsAdded: number;
    totalTensorsSnapshot: number;
    inputShapes: number[][];
    outputShapes: number[][];
    kernelTimeMs: number | {error: string} | Promise<number|{error: string}>;
    extraInfo: string | Promise<string>;
};

export type Result = {
    endIndex: number;
    score: number;
    startIndex: number;
    text: string;
}