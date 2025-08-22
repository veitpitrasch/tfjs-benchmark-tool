import React, {useState} from 'react';
import { TfBackend } from '../types';
import * as tf from '@tensorflow/tfjs';
import {Button, Container} from '@mui/material';
import {AggregationTable} from './AggregationTable';
import {ProfileInfo} from '../types';
import makeStyles from '@mui/styles/makeStyles';

const useStyles = makeStyles({
    buttonContainer: {
        display: 'flex',
        flexDirection: 'row',
        justifyContent: 'space-between',
        gap: '1rem',
    },
    container: {
        display: 'flex',
        flexDirection: 'column',
        gap: '1rem',
    }
});

interface IResNetProps {
    backend: TfBackend;
    warmUpRounds: number;
    epochRounds: number;
}

export const Operations: React.FunctionComponent<IResNetProps> = (props) => {
    const classes = useStyles();
    const [profile, setProfile] = useState<ProfileInfo | null>(null);
    const [timeAverage, setTimeAverage] = useState<number | null>(null);
    const [isRunning, setIsRunning] = useState<boolean>(false);
    const batch = 1;
    const height = 256;
    const width = 256;
    const inChannels = 3;
    const outChannels = 16;
    const filterSize = 7;

    const setTensor = async () => {
        return tf.randomNormal([10, 1024, 1024]);
    };

    const matMul = async () => {
        setIsRunning(true);
        await tf.setBackend(props.backend);
        await tf.ready();
        const a = await setTensor();
        const b = await setTensor();
        let times = [];
        await warmupOperation(() => tf.matMul(a, b));
        for (let i = 0; i < props.epochRounds; i++) {
            const start = performance.now();
            const profile = await tf.profile(() => {
                return tf.matMul(a, b);
            });
            const end = performance.now();
            times.push(end - start);
            if (i === props.epochRounds - 1) {
                setProfile({ ...profile });
            }
            if (profile.result instanceof tf.Tensor) {
                profile.result.dispose();
            }
        }
        const avgTime = times.reduce((sum, t) => sum + t, 0) / times.length;
        setTimeAverage(avgTime);
        tf.disposeVariables();
        tf.engine().disposeVariables();
        tf.engine().reset();
        setIsRunning(false);
    };

    const matAddition = async () => {
        setIsRunning(true);
        await tf.setBackend(props.backend);
        await tf.ready();
        const a = await setTensor();
        const b = await setTensor();
        let times = [];
        await warmupOperation(() => tf.add(a,b));
        for (let i = 0; i < props.epochRounds; i++) {
            const start = performance.now();
            const profile = await tf.profile(() => {
                return tf.add(a, b);
            });
            const end = performance.now();
            times.push(end - start);
            if (i === props.epochRounds - 1) {
                setProfile({ ...profile });
            }
            if (profile.result instanceof tf.Tensor) {
                profile.result.dispose();
            }
        }
        const avgTime = times.reduce((sum, t) => sum + t, 0) / times.length;
        setTimeAverage(avgTime);
        tf.disposeVariables();
        tf.engine().disposeVariables();
        tf.engine().reset();
        setIsRunning(false);
    };

    const conv2d = async () => {
        setIsRunning(true);
        await tf.setBackend(props.backend);
        await tf.ready();
        await tf.conv2d(tf.zeros([1, 8, 8, 3]) as tf.Tensor4D, tf.zeros([3, 3, 3, 8]), 1, 'same').data();
        const input = tf.randomNormal([batch, height, width, inChannels]) as tf.Tensor4D;
        const filter = tf.randomNormal([filterSize, filterSize, inChannels, outChannels]) as tf.Tensor4D;
        let times = [];
        await warmupOperation(() => tf.conv2d(input, filter, 1, 'valid'));
        for (let i = 0; i < props.epochRounds; i++) {
            const start = performance.now();
            const profile = await tf.profile(() => {
                return tf.conv2d(input, filter, 1, 'valid');
            });
            const end = performance.now();
            times.push(end - start);
            if (i === props.epochRounds - 1) {
                setProfile({ ...profile });
            }
            if (profile.result instanceof tf.Tensor) {
                profile.result.dispose();
            }
        }
        const avgTime = times.reduce((sum, t) => sum + t, 0) / times.length;
        setTimeAverage(avgTime);
        input.dispose();
        filter.dispose();
        tf.disposeVariables();
        tf.engine().disposeVariables();
        tf.engine().reset();
        setIsRunning(false);
    };

    const softMax = async () => {
        setIsRunning(true);
        await tf.setBackend(props.backend);
        await tf.ready();
        const a = await setTensor();
        let times = [];
        await warmupOperation(() => tf.softmax(a));
        for (let i = 0; i < props.epochRounds; i++) {
            const start = performance.now();
            const profile = await tf.profile(() => {
                return tf.softmax(a);
            });
            const end = performance.now();
            times.push(end - start);
            if (i === props.epochRounds - 1) {
                setProfile({ ...profile });
            }
            if (profile.result instanceof tf.Tensor) {
                profile.result.dispose();
            }
        }
        const avgTime = times.reduce((sum, t) => sum + t, 0) / times.length;
        setTimeAverage(avgTime);
        tf.disposeVariables();
        tf.engine().disposeVariables();
        tf.engine().reset();
        setIsRunning(false);
    };

    const sigmoid = async () => {
        setIsRunning(true);
        await tf.setBackend(props.backend);
        await tf.ready();
        const a = await setTensor();
        let times = [];
        await warmupOperation(() => tf.sigmoid(a));
        for (let i = 0; i < props.epochRounds; i++) {
            const start = performance.now();
            const profile = await tf.profile(() => {
                return tf.sigmoid(a);
            });
            const end = performance.now();
            times.push(end - start);
            if (i === props.epochRounds - 1) {
                setProfile({ ...profile });
            }
            if (profile.result instanceof tf.Tensor) {
                profile.result.dispose();
            }
        }
        const avgTime = times.reduce((sum, t) => sum + t, 0) / times.length;
        setTimeAverage(avgTime);
        tf.disposeVariables();
        tf.engine().disposeVariables();
        tf.engine().reset();
        setIsRunning(false);
    };

    const warmupOperation = async (op: () => tf.Tensor | Promise<tf.Tensor>, runs = props.warmUpRounds) => {
        for (let i = 0; i < runs; i++) {
            const profile = await tf.profile(() => op());
            if (profile.result instanceof tf.Tensor) profile.result.dispose();
        }
    };

    return (
        <Container maxWidth='sm' className={classes.container}>
            <h3>Common ML operations</h3>
            <div className={classes.buttonContainer}>
                <Button variant='outlined' onClick={matMul} disabled={isRunning}>
                    Run MatMul
                </Button>
                <Button variant='outlined' onClick={matAddition} disabled={isRunning}>
                    Run MatAddition
                </Button>
                <Button variant='outlined' onClick={conv2d} disabled={isRunning}>
                    Run conv2d
                </Button>
                <Button variant='outlined' onClick={softMax} disabled={isRunning}>
                    Run softMax
                </Button>
                <Button variant='outlined' onClick={sigmoid} disabled={isRunning}>
                    Run Sigmoid
                </Button>
            </div>
            <AggregationTable profileInfo={profile} timeAverage={timeAverage}/>
        </Container>
    );
};
