import React, {useCallback, useState} from 'react';
import makeStyles from '@mui/styles/makeStyles';
import {Container, Stack, Typography} from '@mui/material';
import clsx from 'clsx';
import {ModelType, TfBackend} from '../types';
import {RNN} from './RNN';
import {MobileNet} from './MobileNet';
import {NLP} from './NLP';
import {Operations} from './Operations';

interface IModelSelectionProps {
    backend: TfBackend;
    warmupRounds: number;
    epochRounds: number;
    trainingRounds: number;
}

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
        '&:focus-within': {
            outline: '2px solid #1976d2',
            outlineOffset: '2px',
        }
    },
    inputSpanSelected: {
        backgroundColor: '#1976d2',
        color: 'white',
    },
    input: {
        position: 'absolute',
        opacity: 0,
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
        border: '2px solid lightgray',
        borderRadius: '1rem',
        padding: '1rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '1rem',
        boxSizing: 'border-box',
    },
    h2: {
        fontSize: '10rem',
    },
    inputContainer: {
        display: 'flex',
        flexDirection: 'row',
        gap: '1rem',
    }
});

export const ModelSelection: React.FunctionComponent<IModelSelectionProps> = (props) => {
    const classes = useStyles();
    const [model, setModel] = useState<ModelType>('rnn');
    const [epochCount, setEpochCount] = useState<number>(0);

    const increaseEpochCount = useCallback(() => {
        setEpochCount(prevState => prevState + 1);
    }, [] );

    return (
        <Container className={classes.container}>
            <Typography variant='h2' fontSize='1rem' fontWeight='bold'>
                Tensorflow.js model
            </Typography>
            <Stack direction='column' spacing={2}>
                <div className={classes.inputContainer}>
                    <label
                        className={clsx(classes.inputSpan, {
                            [classes.inputSpanSelected]: model === 'rnn',
                        })}
                    >
                        <input
                            type='radio'
                            value='rnn'
                            name='model'
                            className={classes.input}
                            onChange={() => setModel('rnn')}
                            checked={model === 'rnn'}
                        />
                        <span className={classes.label}>RNN</span>
                    </label>
                    <label
                        className={clsx(classes.inputSpan, {
                            [classes.inputSpanSelected]: model === 'mobileNetV3',
                        })}
                    >
                        <input
                            type='radio'
                            value='mobileNetV3'
                            name='model'
                            className={classes.input}
                            onChange={() => setModel('mobileNetV3')}
                            checked={model === 'mobileNetV3'}
                        />
                        <span className={classes.label}>MobileNetV3</span>
                    </label>
                    <label
                        className={clsx(classes.inputSpan, {
                            [classes.inputSpanSelected]: model === 'nlp',
                        })}
                    >
                        <input
                            type='radio'
                            value='nlp'
                            name='model'
                            className={classes.input}
                            onChange={() => setModel('nlp')}
                            checked={model === 'nlp'}
                        />
                        <span className={classes.label}>NLP</span>
                    </label>
                    <label
                        className={clsx(classes.inputSpan, {
                            [classes.inputSpanSelected]: model === 'matmulti',
                        })}
                    >
                        <input
                            type='radio'
                            value='matmulti'
                            name='model'
                            className={classes.input}
                            onChange={() => setModel('matmulti')}
                            checked={model === 'matmulti'}
                        />
                        <span className={classes.label}>Operations</span>
                    </label>
                </div>
                <div>
                    {model === 'rnn' && (
                        <RNN setEpochCount={increaseEpochCount} backend={props.backend} epochRounds={props.epochRounds} trainingRounds={props.trainingRounds} warmupRounds={props.warmupRounds} />
                    )}
                    {model === 'mobileNetV3' && (
                        <MobileNet backend={props.backend} epochRounds={props.epochRounds} warmUpRounds={props.warmupRounds} />
                    )}
                    {model === 'nlp' && (
                        <NLP backend={props.backend} epochRounds={props.epochRounds} />
                    )}
                    {model === 'matmulti' && (
                        <Operations backend={props.backend} warmUpRounds={props.warmupRounds} epochRounds={props.epochRounds}/>
                    )}
                </div>
            </Stack>
        </Container>
    )
}