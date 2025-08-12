import React, {useEffect, useRef, useState} from 'react';
import {Button, Box, TextField, Container} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import '@tensorflow/tfjs-backend-wasm'
import {ModelRegistry, SEQ_LENGTH} from '../models';
import * as tfvis from '@tensorflow/tfjs-vis';
import {ProfileInfo, TfBackend, TrainingText} from '../types';
import makeStyles from '@mui/styles/makeStyles';
import {AggregationTable} from './AggregationTable';

const useStyles = makeStyles({
    generatedText: {
        width: '100%',
        boxSizing: 'border-box',
        padding: '2rem 0',
    },
    textField: {
        width: '50%',
    },
    container: {
        display: 'flex',
        flexDirection: 'column',
        gap: '1rem',
    },
    buttonContainer: {
        display: 'flex',
        flexDirection: 'row',
        gap: '1rem',
    }
});

interface IRNNProps {
    setEpochCount: () => void;
    epochRounds: number;
    backend: TfBackend;
    trainingRounds: number;
    warmupRounds: number;
}

const PREDEFINED_CHARSET = Array.from("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789äÄöÖüÜ/ßé»« {}[].,!?;:|'\"-()\n");

export const RNN: React.FC<IRNNProps> = (props) => {
    const classes = useStyles();
    const inputRef = useRef<HTMLTextAreaElement>(null);
    const [generatedText, setGeneratedText] = useState('');
    const modelRef = useRef<tf.LayersModel | null>(null);
    const charSetRef = useRef<string[]>([]);
    const [model, setModel] = useState<tf.LayersModel | null>(null);
    const [isTrained, setIsTrained] = useState(false);
    const [epochCount, setEpochCount] = useState(0);
    const [trainingText, setTrainingText] = useState<string>('');
    const [totalTrainingTime, setTotalTrainingTime] = useState<number>();
    const [profile, setProfile] = useState<ProfileInfo | null>(null);
    const [timeAverage, setTimeAverage] = useState<number | null>(null);
    const [initializationTime, setInitializationTime] = useState<number | null>(null);

    useEffect(() => {
        loadText('hamlet');
    }, []);

    const loadText = (name: TrainingText) => {
        fetch(`/${name}.txt`)
            .then(res => res.text())
            .then(text => {
                setTrainingText(text);
            })
            .catch(err => console.log('Unable to load txt file: ', err))
    }

    const initializeModel = async () => {
        await tf.setBackend(props.backend);
        await tf.ready();
        const start = performance.now();
        const model = ModelRegistry.rnn.create(PREDEFINED_CHARSET.length);
        const end = performance.now();
        const surface = { name: 'Layer Summary', tab: 'Model Inspection'};
        await tfvis.show.layer(surface, model.getLayer('', 1));
        const inputShape = [1, SEQ_LENGTH]
        const warmupResult = model.predict(tf.zeros(inputShape)) as tf.Tensor;
        warmupResult.dataSync();
        warmupResult.dispose();
        modelRef.current = model;
        setInitializationTime(end - start);
        setModel(model);
    }

    const train = async () => {
        await tf.setBackend(props.backend);
        await tf.ready();
        const text = inputRef.current?.value ?? '';
        if (text.length < 10) {
            return;
        }
        charSetRef.current = PREDEFINED_CHARSET;
        const charToIdx = Object.fromEntries(PREDEFINED_CHARSET.map((c, i) => [c, i]));
        const invalidChars = [...text].filter(c => !PREDEFINED_CHARSET.includes(c));
        if (invalidChars.length > 0) {
            return;
        }
        const seqLength = 10;
        const xs: number[][] = [];
        const ys: number[][] = [];
        for (let i = 0; i < text.length - seqLength; i++) {
            const inputSeq = text.slice(i, i + seqLength);
            const targetChar = text[i + seqLength];
            xs.push([...inputSeq].map(c => charToIdx[c]));
            const y = new Array(PREDEFINED_CHARSET.length).fill(0);
            y[charToIdx[targetChar]] = 1;
            ys.push(y);
        }
        const xsTensor = tf.tensor2d(xs, [xs.length, seqLength]);
        const ysTensor = tf.tensor2d(ys, [ys.length, PREDEFINED_CHARSET.length]);
        if (model) {
            const start = performance.now();
            const history = await model.fit(xsTensor, ysTensor, {
                epochs: props.trainingRounds,
                batchSize: 32,
                callbacks: [
                    {
                        onEpochEnd: (epoch: number, logs?: tf.Logs) => {
                            setEpochCount(prev => prev + 1);
                            props.setEpochCount();
                        },
                    },
                    tfvis.show.fitCallbacks(
                        { name: 'Training Performance' },
                        ['loss', 'mse'],
                        { height: 200, callbacks: ['onEpochEnd'] }
                    )
                ]
            });
            const end = performance.now();
            setTotalTrainingTime(end - start);
        }
        xsTensor.dispose();
        ysTensor.dispose();
        setIsTrained(true);
    }

    const generate = async () => {
        console.log(tf.getBackend());
        const charSet = charSetRef.current;
        if (!model || charSet.length === 0) {
            return;
        }
        const charToIdx = Object.fromEntries(charSet.map((c, i) => [c, i]));
        const idxToChar = Object.fromEntries(charSet.map((c, i) => [i, c]));
        let seed = (inputRef.current?.value ?? '').slice(0, 10);
        let result = seed;
        let times = [];
        for (let w = 0; w < props.warmupRounds; w++) {
            const inputIndices = [...seed].map(c => charToIdx[c] ?? 0);
            const inputTensor = tf.tensor2d([inputIndices], [1, inputIndices.length]);
            const prediction = model.predict(inputTensor) as tf.Tensor;
            const probs = (prediction.arraySync() as number[][])[0];
            const nextIdx = sample(probs);
            const nextChar = idxToChar[nextIdx] ?? '';
            seed = (result + nextChar).slice(-10);
            tf.dispose([inputTensor, prediction]);
        }
        for (let i = 0; i < props.epochRounds; i++) {
            const inputIndices = [...seed].map(c => charToIdx[c] ?? 0);
            const inputTensor = tf.tensor2d([inputIndices], [1, inputIndices.length]);
            const start = performance.now();
            const profile = await tf.profile(() => {
                return model.predict(inputTensor);
            });
            const end = performance.now();
            times.push(end - start);
            if (i === props.epochRounds - 1) {
                setProfile({...profile});
            }
            const prediction = model.predict(inputTensor) as tf.Tensor;
            const probs = (prediction.arraySync() as number[][])[0];

            const nextIdx = sample(probs);
            const nextChar = idxToChar[nextIdx] ?? '';
            result += nextChar;
            seed = result.slice(-10);
            tf.dispose([inputTensor, prediction]);
            setGeneratedText(prev => prev + nextChar);
        }
        const avgTime = times.reduce((sum, t) => sum + t, 0) / times.length;
        setTimeAverage(avgTime);
    };

    const sample = (probs: number[]) => {
        const r = Math.random();
        let acc = 0;
        for (let i = 0; i < probs.length; i++) {
            acc += probs[i];
            if (r < acc) return i;
        }
        return probs.length - 1;
    };

    return (
        <Container maxWidth='sm'>
            <h3>Text Generation with a RNN</h3>
            <div className={classes.container}>
                <Box>
                    <Button onClick={initializeModel} variant='outlined'>
                        Initialize model
                    </Button>
                </Box>
                <Box sx={{ flexDirection: 'column' }}>
                    <TextField
                        className={classes.textField}
                        multiline={true}
                        minRows={5}
                        maxRows={10}
                        inputRef={inputRef}
                        value={trainingText}
                        fullWidth={true}
                        label='Train Model'
                    />
                </Box>
                <div className={classes.buttonContainer}>
                    <Button onClick={train} disabled={!model} variant='outlined'>
                        Train model
                    </Button>
                    <Button
                        endIcon={<SendIcon/>}
                        variant='outlined'
                        onClick={generate}
                        disabled={!model || !isTrained}
                    >
                        Generate Text
                    </Button>
                </div>
                {generatedText && (
                    <div className={classes.generatedText}>{generatedText}</div>
                )}
            </div>
            <AggregationTable profileInfo={profile} timeAverage={timeAverage} trainingTime={totalTrainingTime} initializationTime={initializationTime} />
        </Container>
    );
};
