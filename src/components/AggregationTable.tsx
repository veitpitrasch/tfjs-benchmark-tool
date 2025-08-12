import React, {useEffect, useState} from "react";
import {Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow} from "@mui/material";
import makeStyles from "@mui/styles/makeStyles";
import {ProfileInfo} from "../types";

const useStyles = makeStyles({
    tableHeader: {
        backgroundColor: 'lightgray',
    },
    tableContainer: {
        boxSizing: 'border-box',
        maxWidth: '1000px',
        display: 'flex',
        flexDirection: 'column',
        gap: '1rem',
    }
});

export type AggregationResult = {
    kernel: string;
    timeMs: number;
}

export type Memory = {
    peakBytes: number;
    newBytes: number;
}

interface IAggregationTableProps {
    profileInfo?: ProfileInfo | null;
    timeAverage?: number | null;
    trainingTime?: number | null;
    initializationTime?: number | null;
}

export const AggregationTable: React.FunctionComponent<IAggregationTableProps> = (props) => {
    const classes = useStyles();
    const [aggregatedTime, setAggregatedTime] = useState<AggregationResult[] | null>(null);
    const [memory, setMemory] = useState<Memory | null>(null);

    useEffect(() => {
        if (!props.profileInfo) {
            return;
        }
        const loadAndProcessProfileInfo = async () => {
            if (!props.profileInfo) {
                return;
            }
            const extraInfos: string[] = await Promise.all(
                props.profileInfo.kernels.map(async k => await k.extraInfo ?? "")
            );
            const timings = extraInfos
                .filter(Boolean)
                .flatMap(str => str.split(",").map(s => s.trim()))
                .map(entry => {
                    const [name, timeStr] = entry.split(":").map(s => s.trim());
                    return {
                        kernel: name,
                        timeMs: parseFloat(timeStr)
                    };
                })
                .filter(item => !isNaN(item.timeMs));
            const totalTime = timings.reduce((sum, { timeMs }) => sum + timeMs, 0);

            const aggregated = Object.values(
                timings.reduce((acc, { kernel, timeMs }) => {
                    if (!acc[kernel]) {
                        acc[kernel] = { kernel, timeMs: 0 };
                    }
                    acc[kernel].timeMs += timeMs;
                    return acc;
                }, {} as Record<string, { kernel: string; timeMs: number }>)
            ).sort((a, b) => b.timeMs - a.timeMs);
            setAggregatedTime([{ kernel: 'Total', timeMs: totalTime }, ...aggregated]);
        };

        loadAndProcessProfileInfo();
        const memory: Memory = {
            peakBytes: props.profileInfo.peakBytes / (1024 * 1024),
            newBytes: props.profileInfo.newBytes / (1024 * 1024),
        };
        setMemory(memory);
    }, [props.profileInfo]);

    if (aggregatedTime) {
        return (
            <div className={classes.tableContainer}>
                <TableContainer component={Paper}>
                    <Table>
                        <TableHead className={classes.tableHeader}>
                            <TableRow>
                                <TableCell>Kernel</TableCell>
                                <TableCell align="right">Time (ms)</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {aggregatedTime && aggregatedTime.map((aggregatedTime, index) => (
                                <TableRow key={index}>
                                    <TableCell>{aggregatedTime.kernel}</TableCell>
                                    <TableCell align="right">{aggregatedTime.timeMs.toFixed(2)}</TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </TableContainer>
                <TableContainer component={Paper}>
                    <Table>
                        <TableHead className={classes.tableHeader}>
                            <TableRow>
                                <TableCell>Memory</TableCell>
                                <TableCell align="right">in MB</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            <TableRow>
                                <TableCell>PeakBytes</TableCell>
                                <TableCell align="right">{memory?.peakBytes.toFixed(2)}</TableCell>
                            </TableRow>
                            <TableRow>
                                <TableCell>NewBytes</TableCell>
                                <TableCell align="right">{memory?.newBytes.toFixed(2)}</TableCell>
                            </TableRow>
                        </TableBody>
                    </Table>
                </TableContainer>
                {props.trainingTime && (
                    <p>
                        Training Time: {props.trainingTime.toFixed(2)} ms
                    </p>
                )}
                <p>
                    Average execution time (with overhead): {props.timeAverage?.toFixed(2)} ms
                </p>
            </div>
        )
    }

    if (props.trainingTime) {
        return (
            <p>
                Training Time: {props.trainingTime.toFixed(2)} ms
            </p>
        )
    }

    if (props.initializationTime) {
        return (
            <p>
                Initialization Time: {props.initializationTime.toFixed(2)} ms
            </p>
        )
    }

    return null;
}