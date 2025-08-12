import React from 'react';
import {Container, Stack, Typography} from '@mui/material';
import makeStyles from '@mui/styles/makeStyles';

const useStyles = makeStyles({
    container: {
        border: '2px solid lightgray',
        borderRadius: '1rem',
        padding: '1rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '1rem',
        boxSizing: 'border-box',
    },
});

interface IModelOptionsProps {
    warmUpRounds: number;
    setWarmUpRounds: (warmUpRounds: number) => void;
    epochRounds: number;
    setEpochRounds: (epochRounds: number) => void;
    trainingRounds: number;
    setTrainingRounds: (trainingRounds: number) => void;
}
export const ModelOptions:React.FunctionComponent<IModelOptionsProps> = (props) => {
    const classes = useStyles();

    const onChangeWarmupRoundsHandler = (event: any) => {
        props.setWarmUpRounds(event.target.value);
    }

    const onChangeEpochRoundsHandler = (event: any) => {
        props.setEpochRounds(event.target.value);
    }

    const onChangeTrainingRoundsHandler = (event: any) => {
        props.setTrainingRounds(event.target.value);
    }

    return (
        <Container className={classes.container}>
            <Typography variant='h2' fontSize='1rem' fontWeight='bold'>
                Model Options
            </Typography>
            <Stack direction='row' spacing={2}>
                <label htmlFor='warmupRounds'>Warm-up rounds</label>
                <input type='number' id='warmupRounds' name='warmupRounds' value={props.warmUpRounds} onChange={onChangeWarmupRoundsHandler}/>
            </Stack>
            <Stack direction='row' spacing={2}>
                <label htmlFor='epochRounds'>Epoch rounds</label>
                <input type='number' id='epochRounds' name='epochRounds' value={props.epochRounds} onChange={onChangeEpochRoundsHandler}/>
            </Stack>
            <Stack direction='row' spacing={2}>
                <label htmlFor='trainingRounds'>Training Rounds (RNN)</label>
                <input type='number' id='trainingRounds' name='trainingRounds' value={props.trainingRounds} onChange={onChangeTrainingRoundsHandler}/>
            </Stack>
        </Container>
    )
}