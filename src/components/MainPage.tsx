import React, {useState} from 'react';
import {Container, Stack} from '@mui/material';
import makeStyles from '@mui/styles/makeStyles';
import {TfBackend} from '../types';
import {BackendSelection} from './BackendSelection';
import {ModelSelection} from './ModelSelection';
import {ModelOptions} from './ModelOptions';

const useStyles = makeStyles({
    inputSpan: {
        backgroundColor: 'rgba(0, 0, 0, 0.08)',
        borderRadius: '1rem',
        display: 'flex',
        alignItems: 'center',
        height: '30px',
        '&:hover': {
            cursor: 'pointer',
        },
    },
    inputSpanSelected: {
        backgroundColor: '#1976d2',
        color: 'white',
    },
    input: {
        display: 'none',
        margin: 'auto',
    },
    label: {
        padding: '0 0.75rem',
        fontSize: '0.8125rem',
        '&:hover': {
            cursor: 'pointer',
        }
    },
    test: {
        height: '100%',
        width: '100%',
    },
    container: {
        gap: '1rem',
    }
});

export const MainPage = () => {
    const classes = useStyles();
    const [backend, setBackend] = useState<TfBackend>('webgpu');
    const [warmUpRounds, setWarmUpRounds] = useState<number>(1);
    const [epochRounds, setEpochRounds] = useState<number>(50);
    const [trainingRounds, setTrainingRounds] = useState<number>(30);

    return (
        <Container className={classes.container}>
            <Stack spacing={3}>
                <h1>Tensorflow.js Benchmark Test</h1>
                <BackendSelection
                    setBackend={setBackend}
                    backend={backend}
                />
                <ModelOptions
                    warmUpRounds={warmUpRounds}
                    setWarmUpRounds={setWarmUpRounds}
                    epochRounds={epochRounds}
                    setEpochRounds={setEpochRounds}
                    trainingRounds={trainingRounds}
                    setTrainingRounds={setTrainingRounds}
                />
                <ModelSelection
                    backend={backend}
                    warmupRounds={warmUpRounds}
                    epochRounds={epochRounds}
                    trainingRounds={trainingRounds}
                />
            </Stack>
        </Container>
    )
}