import React from 'react';
import {Container, Stack, Typography} from '@mui/material';
import clsx from 'clsx';
import makeStyles from '@mui/styles/makeStyles';
import {TfBackend} from '../types';

interface IBackendSelectionProps {
    setBackend: (backend: TfBackend) => void;
    backend: TfBackend;
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
    },
    h2: {
        fontSize: '10rem',
    }
});

export const BackendSelection: React.FunctionComponent<IBackendSelectionProps> = (props) => {
    const classes = useStyles();

    return (
        <Container className={classes.container}>
            <Typography variant='h2' fontSize='1rem' fontWeight='bold'>
                Tensorflow.js backend
            </Typography>
            <Stack direction='row' spacing={2}>
                <span className={clsx({
                    [classes.inputSpan]: true,
                    [classes.inputSpanSelected]: props.backend === 'cpu',
                })}>
                    <span className={classes.test}>
                        <input type='radio' id='cpu' value='cpu' name='backend' className={classes.input} onClick={() => props.setBackend('cpu')}/>
                    </span>
                        <label htmlFor='cpu' className={classes.label}>cpu</label>
                </span>
                <span className={clsx({
                    [classes.inputSpan]: true,
                    [classes.inputSpanSelected]: props.backend === 'webgl',
                })}>
                    <input type='radio' id='webgl' value='webgl' name='backend' className={classes.input} onClick={() => props.setBackend('webgl')}/>
                    <label htmlFor='webgl' className={classes.label}>webgl</label>
                </span>
                <span className={clsx({
                    [classes.inputSpan]: true,
                    [classes.inputSpanSelected]: props.backend === 'webgpu',
                })}>
                    <input type='radio' id='webgpu' value='webgpu' name='backend' className={classes.input} onClick={() => props.setBackend('webgpu')}/>
                    <label htmlFor='webgpu' className={classes.label}>webgpu</label>
                </span>
            </Stack>
        </Container>
    )
}