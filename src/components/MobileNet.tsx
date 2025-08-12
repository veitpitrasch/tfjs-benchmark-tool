import React, {useState, useRef} from 'react';
import * as mobilenet from '@tensorflow-models/mobilenet';
import {ProfileInfo, TfBackend} from '../types';
import clsx from 'clsx';
import {
    Button,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Paper,
    Container
} from '@mui/material';
import * as tf from '@tensorflow/tfjs';
import makeStyles from '@mui/styles/makeStyles';
import {AggregationTable} from './AggregationTable';

const useStyles = makeStyles({
    tableHeader: {
        backgroundColor: 'lightgray',
    },
    img: {
        boxSizing: 'border-box',
        cursor: 'pointer',
        border: '3px solid transparent',
        height: '100px',
    },
    selectedImg: {
        borderColor: 'blue',
    },
    buttonContainer: {
        display: 'flex',
        flexDirection: 'row',
        gap: '1rem',
    },
    tableContainer: {
        display: 'flex',
        flexDirection: 'column',
        gap: '1rem',
    },
    container: {
        display: 'flex',
        flexDirection: 'column',
        gap: '1rem',
    },
    imageInput: {
        position: 'absolute',
        opacity: 0,
    },

    imageLabel: {
        display: 'inline-block',
        marginRight: '12px',
        '&:focus-within img': {
            outline: '3px solid #1976d2',
            outlineOffset: '2px',
        }
    },
});


const imageOptions = [
    {
        name: 'dog',
        url: './dog.jpg',
    },
    {
        name: 'frog',
        url: './frog.jpg',
    },
    {
        name: 'flower',
        url: './flower.jpg',
    }
]

type ClassificationResult = {
    className: string;
    probability: number;
}
interface IMobileNetProps {
    backend: TfBackend;
    warmUpRounds: number;
    epochRounds: number;
}

export const MobileNet: React.FunctionComponent<IMobileNetProps> = (props) => {
    const classes = useStyles();
    const [model, setModel] = useState<mobilenet.MobileNet | null>(null);
    const [selectedImage, setSelectedImage] = useState<string | null>(null);
    const [predictions, setPredictions] = useState<ClassificationResult[] | null>(null);
    const [initializationTime, setInitializationTime] = useState<number | null>(null);
    const imageRef = useRef<HTMLImageElement>(null);
    const [profile, setProfile] = useState<ProfileInfo | null>(null);
    const [timeAverage, setTimeAverage] = useState<number | null>(null);
    const [isRunning, setIsRunning] = useState<boolean>(false);

    const initializeModel = async () => {
        await tf.setBackend(props.backend);
        await tf.ready();
        const start = performance.now();
        const loadedModel = await mobilenet.load();
        const end = performance.now();
        setInitializationTime(end - start);
        setModel(loadedModel);
    };

    const classify = async () => {
        if (!model || !imageRef.current) return;
        await tf.setBackend(props.backend);
        await tf.ready();
        setIsRunning(true);
        let times = [];
        await warmupOperation();
        for (let i = 0; i < props.epochRounds; i++) {
            const start = performance.now();
            const profile = await tf.profile(() => {
                // @ts-ignore
                return model.classify(imageRef.current);
            });
            const end = performance.now();
            const result = profile.result as ClassificationResult[];
            setPredictions(result);
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
        setIsRunning(false);
    };

    const warmupOperation = async () => {
        if (!imageRef.current) return;
        for (let i = 0; i < props.warmUpRounds; i++) {
            model?.classify(imageRef.current);
        }
    };

    return (
        <Container maxWidth='sm' className={classes.container}>
            <h3>Image classification with MobileNetV3</h3>
            <div>
                {imageOptions.map((img, index) => (
                    <label
                        key={index}
                        className={classes.imageLabel}
                    >
                        <input
                            type='radio'
                            name='image'
                            value={img.url}
                            checked={selectedImage === img.url}
                            onChange={() => setSelectedImage(img.url)}
                            className={classes.imageInput}
                            disabled={isRunning}
                        />
                        <img
                            ref={selectedImage === img.url ? imageRef : null}
                            src={img.url}
                            width={150}
                            className={clsx(classes.img, {
                                [classes.selectedImg]: selectedImage === img.url
                            })}
                            crossOrigin='anonymous'
                            alt={img.name}
                        />
                    </label>
                ))}
            </div>
            <div className={classes.buttonContainer}>
                <Button variant='outlined' onClick={initializeModel} disabled={isRunning}>
                    Initialize model
                </Button>
                <Button
                    onClick={classify}
                    variant='outlined'
                    disabled={!selectedImage || !model || isRunning}
                >
                    Classify Image
                </Button>
            </div>
            <div className={classes.tableContainer}>
                {predictions && predictions.length > 0 && (
                    <TableContainer component={Paper}>
                        <Table>
                            <TableHead className={classes.tableHeader}>
                                <TableRow>
                                    <TableCell>Classname</TableCell>
                                    <TableCell align='right'>Probability (%)</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {predictions.map((prediction, index) => (
                                    <TableRow key={index}>
                                        <TableCell>{prediction.className}</TableCell>
                                        <TableCell align='right'>{(prediction.probability * 100).toFixed(2)}</TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </TableContainer>
                )}
                <AggregationTable profileInfo={profile} timeAverage={timeAverage} initializationTime={initializationTime} />
            </div>
        </Container>
    );
};
