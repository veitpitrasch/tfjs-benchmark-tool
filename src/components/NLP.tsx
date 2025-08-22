import React, {useEffect, useState} from 'react';
import {ProfileInfo, Result, TfBackend} from '../types';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as tf from '@tensorflow/tfjs';
import {
    Button, Container,
    Paper,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    TextField
} from '@mui/material';
import * as qna from '@tensorflow-models/qna';
import makeStyles from '@mui/styles/makeStyles';
import {AggregationTable} from './AggregationTable';

const useStyles = makeStyles({
    container: {
        display: 'flex',
        gap: '1rem',
        flexDirection: 'column',
    },
    tableHeader: {
        backgroundColor: 'lightgray',
    },
    textFieldContainer: {
        display: 'flex',
        flexDirection: 'column',
        gap: '1rem',
    },
    buttonContainer: {
        display: 'flex',
        flexDirection: 'row',
        gap: '1rem',
    },
});

interface INLPProps {
    backend: TfBackend;
    epochRounds: number;
}

export const NLP: React.FunctionComponent<INLPProps> = (props) => {
    const classes = useStyles();
    const [model, setModel] = useState<cocoSsd.ObjectDetection | tf.GraphModel | qna.QuestionAndAnswer | null>(null);
    const [initializationTime, setInitializationTime] = useState<number | null>(null);
    const [predictions, setPredictions] = useState<Result[] | null>(null);
    const [isRunning, setIsRunning] = useState(false);
    const [timeAverage, setTimeAverage] = useState<number | null>(null);
    const [profile, setProfile] = useState<ProfileInfo | null>(null);
    const [question, setQuestion] = useState<string>('Who is the CEO of Google?');
    const [passage, setPassage] = useState<string>('');

    useEffect(() => {
        loadText('google');
    }, []);

    const loadText = (name: string) => {
        fetch(`/${name}.txt`)
            .then(res => res.text())
            .then(text => {
                setPassage(text);
            })
            .catch(err => console.log('Unable to load txt file: ', err))
    }

    const initializeModel = async () => {
        await tf.setBackend(props.backend);
        await tf.ready();
        const start = performance.now();
        const loadedModel = await qna.load();
        const end = performance.now();
        setInitializationTime(end - start);
        setModel(loadedModel);
    };

    const detect = async () => {
        if (!model) return;
        setIsRunning(true);
        await tf.setBackend(props.backend);
        await tf.ready();
        let times = [];
        for (let i = 0; i < props.epochRounds; i++) {
            const start = performance.now();
            const profile = await tf.profile(async () => {
                const timing = await tf.time(() => {
                    // @ts-ignore
                    return model.findAnswers(question, passage);
                });
                console.log(`Kernel: ${timing.kernelMs} ms, Wall: ${timing.wallMs} ms`);
            });
            const end = performance.now();
            times.push(end - start);
            if (i === props.epochRounds - 1) {
                setProfile({...profile});
                const result = profile.result as Result[];
                setPredictions(result);
            }
            if (profile.result instanceof tf.Tensor) {
                profile.result.dispose();
            }
        }
        const avgTime = times.reduce((sum, t) => sum + t, 0) / times.length;
        setTimeAverage(avgTime);
        setIsRunning(false);
    }

    return (
        <Container className={classes.container} maxWidth='sm'>
            <h3>Natural Language Processing with BERT</h3>
            <div className={classes.textFieldContainer}>
                <TextField multiline={true} value={passage} fullWidth={true} label='Passage' />
                <TextField value={question} label='Question' />
            </div>
            <div className={classes.buttonContainer}>
                <Button variant='outlined' onClick={initializeModel} disabled={isRunning}>
                    Initialize model
                </Button>
                <Button variant='outlined' onClick={detect} disabled={!model || isRunning}>
                    Run model
                </Button>
            </div>
            {predictions && predictions.length > 0 && (
                <TableContainer component={Paper}>
                    <Table>
                        <TableHead className={classes.tableHeader}>
                            <TableRow>
                                <TableCell>Text</TableCell>
                                <TableCell align='right'>Score</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {predictions.map((prediction, index) => (
                                <TableRow key={index}>
                                    <TableCell>{prediction.text}</TableCell>
                                    <TableCell align='right'>{(prediction.score * 100).toFixed(2)}</TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </TableContainer>
            )}
            <AggregationTable profileInfo={profile} timeAverage={timeAverage} initializationTime={initializationTime} />
        </Container>
    );
}