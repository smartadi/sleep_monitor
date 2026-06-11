#!/usr/bin/env python
"""
Generate the multi-channel rate estimation section as a standalone Word document.

This is a secondary writeup for review. The user decides whether to merge
it into the main paper.

Output: writeup/CAP_rate_consolidation_section.docx
"""

from __future__ import annotations
import sys
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / 'writeup' / 'figures' / 'rate_consolidation'
OUT = ROOT / 'writeup' / 'CAP_rate_consolidation_section.docx'


def setup_styles(doc: Document):
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(11)
    style.paragraph_format.space_after = Pt(6)
    style.paragraph_format.line_spacing = 1.15

    for level in range(1, 4):
        sname = f'Heading {level}'
        s = doc.styles[sname]
        s.font.name = 'Times New Roman'
        s.font.color.rgb = RGBColor(0, 0, 0)
        if level == 1:
            s.font.size = Pt(14)
            s.font.bold = True
            s.paragraph_format.space_before = Pt(18)
            s.paragraph_format.space_after = Pt(8)
        elif level == 2:
            s.font.size = Pt(12)
            s.font.bold = True
            s.paragraph_format.space_before = Pt(14)
            s.paragraph_format.space_after = Pt(6)
        else:
            s.font.size = Pt(11)
            s.font.bold = True
            s.font.italic = True
            s.paragraph_format.space_before = Pt(10)
            s.paragraph_format.space_after = Pt(4)

    return doc


def add_figure(doc: Document, path: Path, caption: str, width: float = 6.0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    if path.exists():
        run.add_picture(str(path), width=Inches(width))
    else:
        p.add_run(f'[MISSING FIGURE: {path.name}]')

    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
    cap.paragraph_format.space_before = Pt(4)
    cap.paragraph_format.space_after = Pt(12)
    run = cap.add_run(caption)
    run.font.size = Pt(9)
    run.font.italic = True
    run.font.name = 'Times New Roman'


def add_table(doc: Document, df: pd.DataFrame, caption: str):
    cap = doc.add_paragraph()
    run = cap.add_run(caption)
    run.font.size = Pt(9)
    run.font.bold = True
    run.font.italic = True
    cap.paragraph_format.space_after = Pt(4)

    table = doc.add_table(rows=len(df) + 1, cols=len(df.columns))
    table.style = 'Light Shading'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for j, col in enumerate(df.columns):
        cell = table.rows[0].cells[j]
        cell.text = col
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                run.font.size = Pt(8)
                run.font.bold = True
                run.font.name = 'Times New Roman'

    for row_idx, (i, row) in enumerate(df.iterrows()):
        for j, val in enumerate(row):
            cell = table.rows[row_idx + 1].cells[j]
            if isinstance(val, float):
                cell.text = f'{val:.1f}' if abs(val) >= 0.1 else f'{val:.2f}'
            else:
                cell.text = str(val)
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(8)
                    run.font.name = 'Times New Roman'

    doc.add_paragraph()


# ==============================================================================
# METHODS
# ==============================================================================

def write_methods(doc: Document):
    doc.add_heading('Methods', level=1)
    doc.add_heading('Multi-Channel Rate Estimation Without PSG Calibration', level=2)

    doc.add_paragraph(
        'The k-factor rate estimation pipeline described previously requires a '
        'per-session calibration step using PSG ground truth. This is adequate '
        'for laboratory validation but presents a problem for deployment: if the '
        'device needs PSG to calibrate itself, it cannot function independently. '
        'We therefore evaluated whether acceptable rate accuracy could be achieved '
        'without any PSG-derived correction, using only the raw sensor signals and '
        'a multi-channel fusion strategy.'
    )

    doc.add_heading('Method and Channel Benchmark', level=3)
    doc.add_paragraph(
        'Five rate estimation methods were evaluated on five channel '
        'configurations across all 12 overnight recordings. The methods were: '
        '(1) Welch periodogram peak frequency, (2) autocorrelation function '
        '(ACF) lag at the first prominent peak, (3) Hilbert instantaneous '
        'frequency, (4) zero-crossing rate, and (5) local maxima counting. The '
        'channel configurations were the three individual sensors (CLE, CRE, CH), '
        'their arithmetic average (Avg = (CLE + CRE) / 2), and a left-right '
        'differential (Diff = CLE - CRE). Each method was applied to '
        'non-overlapping 30-second windows in both the respiratory (0.1-0.5 Hz) '
        'and cardiac (0.7-4.0 Hz) bands. No k-factor scaling was applied; the '
        'raw estimated rate was compared directly against PSG ground truth.'
    )

    doc.add_heading('Channel Fusion', level=3)
    doc.add_paragraph(
        'Two fusion strategies were tested to combine information across '
        'channels. In confidence-weighted fusion, each channel contributes its '
        'rate estimate weighted by a composite signal quality score (derived from '
        'signal-to-noise ratio, spectral concentration, and autocorrelation '
        'prominence). In agreement-filtered fusion, channels are first screened '
        'for cross-method consistency: only channels where the standard deviation '
        'across the five methods falls below a threshold (0.15 Hz for cardiac, '
        '0.05 Hz for respiratory) are included, then confidence-weighted as '
        'above. An oracle selector, which picks the best channel per window with '
        'hindsight knowledge of the ground truth, establishes an upper bound on '
        'what any per-window channel selection could achieve.'
    )

    doc.add_heading('Continuous Wavelet Transform Ridge Tracking', level=3)
    doc.add_paragraph(
        'For the cardiac band specifically, we added a continuous wavelet '
        'transform (CWT) method. The CWT scalogram was computed using a Morlet '
        'wavelet over 32 logarithmically spaced scales spanning the cardiac band. '
        'The dominant ridge -- the scale with maximum energy -- was extracted per '
        'window and converted to a frequency estimate. The motivation was that '
        'CWT ridge tracking may resolve the fundamental cardiac repetition rate '
        'directly from the time-frequency representation, without the harmonic '
        'confusion that affects peak-based and Hilbert methods on BCG signals.'
    )

    doc.add_heading('Temporal Smoothing', level=3)
    doc.add_paragraph(
        'All per-window rate estimates were post-processed with a Viterbi-style '
        'temporal smoother. The algorithm finds the path through a discrete grid '
        'of candidate rates that minimizes a cost function combining fidelity to '
        'the observed per-window estimate and a penalty for epoch-to-epoch jumps '
        'exceeding physiological limits (5 BPM/epoch for cardiac, 2 br/min/epoch '
        'for respiratory). This enforces the physiological constraint that heart '
        'rate and breathing rate change gradually, suppressing isolated outlier '
        'windows that would otherwise dominate the error.'
    )


# ==============================================================================
# RESULTS
# ==============================================================================

def write_results(doc: Document):
    doc.add_heading('Results', level=1)
    doc.add_heading('Multi-Channel Rate Estimation', level=2)

    # --- Method benchmark ---
    doc.add_heading('Method Benchmark', level=3)
    doc.add_paragraph(
        'Figure 1 shows the median absolute error (MAE) for every combination '
        'of method and channel, without k-factor scaling, pooled across all 12 '
        'sessions.'
    )
    doc.add_paragraph(
        'For respiratory rate, the spectral (Welch periodogram) method was the '
        'clear winner at 1.5 br/min MAE, nearly uniform across all five channel '
        'configurations. This is notable because it outperforms the k-scaled peak '
        'counting method used in the previous analysis (2.20 br/min). The '
        'spectral method does not count individual breath events; it identifies '
        'the dominant frequency in the power spectrum, which sidesteps the '
        'overcounting problem that necessitated k-factor correction in the first '
        'place. The Hilbert method (2.0 br/min on the best channel) and peak '
        'counting (2.5 br/min) were worse, and ACF was the poorest at '
        '4.6-5.1 br/min.'
    )
    doc.add_paragraph(
        'For cardiac rate, no method performed well. The best was ACF at '
        '12.5 BPM MAE (Avg channel), followed by spectral at 15.5 BPM. Hilbert '
        '(38 BPM), zero-crossing (43-48 BPM), and peak counting (46-51 BPM) '
        'were all far worse -- these methods are severely misled by the '
        'multi-component structure of the BCG waveform, which contains systolic '
        'and dicrotic peaks per cardiac cycle. Without k-factor correction, '
        'every method that counts events or tracks instantaneous phase is '
        'approximately doubling the true rate.'
    )

    add_figure(
        doc,
        FIGURES / 'phase1_method_channel_heatmap.png',
        'Figure 1. Median absolute error for five estimation methods across '
        'five channel configurations, pooled over all 12 sessions. No k-factor '
        'scaling applied. Left: respiratory band (best: spectral, 1.5 br/min). '
        'Right: cardiac band (best: ACF, 12.5 BPM). The cardiac errors are '
        'substantially larger because every method except spectral and ACF '
        'double-counts the multi-component BCG waveform.',
        width=6.2
    )

    # --- Channel fusion ---
    doc.add_heading('Channel Fusion', level=3)
    doc.add_paragraph(
        'For respiratory rate, channel fusion provided no benefit (Figure 2, '
        'left). All four strategies -- best fixed channel, confidence-weighted '
        'fusion, agreement-filtered fusion, and oracle -- produced the same '
        '1.5 br/min MAE. The respiratory signal is sufficiently uniform across '
        'sensor locations that combining channels adds nothing. The oracle and '
        'the worst strategy are identical, meaning that for breathing rate, it '
        'does not matter which channel you use.'
    )
    doc.add_paragraph(
        'For cardiac rate, fusion helped modestly (Figure 2, right). '
        'Confidence-weighted fusion (10.1 BPM) reduced MAE by 19% compared to '
        'the best fixed channel (12.5 BPM). Agreement-filtered fusion '
        '(11.4 BPM) performed worse than pure confidence weighting, suggesting '
        'that the cross-method agreement filter is too aggressive and discards '
        'useful channels. The oracle (4.5 BPM) demonstrates that the information '
        'for accurate cardiac rate estimation exists in the multi-channel array '
        'on a per-window basis -- the problem is selecting the right channel at '
        'the right time. The gap between the oracle and the best fusion strategy '
        'is large (4.5 vs 10.1 BPM), indicating that our quality-based channel '
        'selection captures only a fraction of the available benefit.'
    )

    add_figure(
        doc,
        FIGURES / 'phase2_fusion_comparison.png',
        'Figure 2. Multi-channel fusion strategies compared. Left: respiratory '
        'rate -- all strategies equivalent (1.5 br/min). Right: cardiac rate -- '
        'confidence-weighted fusion (10.1 BPM) improves over best fixed channel '
        '(12.5 BPM), but remains far from the per-window oracle (4.5 BPM).',
        width=6.2
    )

    # --- CWT ---
    doc.add_heading('CWT Ridge Tracking', level=3)
    doc.add_paragraph(
        'The CWT ridge method was the best-performing cardiac rate estimator '
        'without k-factor scaling, achieving 11.6 BPM MAE on the Avg channel '
        '(Figure 3). This is a meaningful improvement over spectral (15.5 BPM) '
        'and a dramatic improvement over Hilbert (38.1 BPM) and peak counting '
        '(49.3 BPM). The CWT method appears to resolve the fundamental cardiac '
        'repetition rate from the scalogram more reliably than methods that '
        'operate in either the time domain (peaks, zero-crossing) or the '
        'frequency domain (spectral, Hilbert) alone.'
    )
    doc.add_paragraph(
        'However, 11.6 BPM is still nearly three times worse than the k-scaled '
        'Hilbert method from the previous analysis (4.19 BPM). The CWT approach '
        'reduces the harmonic confusion problem but does not eliminate it. The '
        'per-channel pattern was consistent: Avg was best, followed by CLE, CH, '
        'CRE, and Diff. The bias was small and positive (1.9 BPM on Avg), '
        'meaning the CWT still slightly overcounts, but far less than the '
        '~38 BPM bias of uncorrected Hilbert.'
    )

    add_figure(
        doc,
        FIGURES / 'phase3_cwt_cardiac.png',
        'Figure 3. Cardiac rate estimation: CWT ridge tracking compared against '
        'Hilbert, spectral, and peak counting, without k-factor scaling. CWT '
        'achieves the lowest MAE on every channel, with the Avg channel best '
        'at 11.6 BPM.',
        width=6.2
    )

    # --- Viterbi ---
    doc.add_heading('Temporal Smoothing', level=3)
    doc.add_paragraph(
        'Viterbi temporal smoothing had no effect on respiratory rate (Figure 4, '
        'left). The respiratory estimates are already smooth because the '
        'spectral method returns stable frequency estimates from window to '
        'window. There is no epoch-to-epoch jitter to suppress.'
    )
    doc.add_paragraph(
        'For cardiac rate, Viterbi smoothing was highly effective, reducing MAE '
        'by 31-35% across all fusion strategies (Figure 4, right). The best '
        'pipeline -- confidence-weighted fusion plus Viterbi -- achieved 6.6 BPM '
        'MAE, down from 10.1 BPM raw. This is the best cardiac accuracy we '
        'obtained without any PSG calibration. The smoothing works because '
        'cardiac rate changes slowly (physiologically bounded at roughly '
        '5 BPM/epoch), so the Viterbi path rejects isolated windows where the '
        'BCG harmonic structure causes a spurious rate doubling or halving.'
    )

    add_figure(
        doc,
        FIGURES / 'phase4_viterbi_improvement.png',
        'Figure 4. Effect of Viterbi temporal smoothing. Left: respiratory rate '
        'unchanged (estimates already stable). Right: cardiac rate reduced by '
        '31-35% across all fusion strategies. Best result: confidence-weighted '
        'fusion + Viterbi = 6.6 BPM MAE.',
        width=6.2
    )

    # --- Bland-Altman ---
    doc.add_heading('Agreement with PSG Ground Truth', level=3)
    doc.add_paragraph(
        'Figure 5 shows Bland-Altman plots for the best pipeline '
        '(confidence-weighted fusion + Viterbi) against PSG ground truth.'
    )
    doc.add_paragraph(
        'For respiratory rate, the mean bias was small (+0.2 br/min) and the '
        '95% limits of agreement were [-5.0, +5.5] br/min. However, the plot '
        'reveals a proportional bias: errors increase with breathing rate, '
        'forming a diagonal fan rather than a horizontal band. At low breathing '
        'rates (8-12 br/min), the CAP estimate tends to overshoot; at higher '
        'rates (18-22 br/min), it undershoots. This pattern likely reflects '
        'the spectral method locking onto a narrower frequency range than the '
        'true breathing rate spans across the night. The practical implication '
        'is that the 1.5 br/min median MAE overstates accuracy at the extremes '
        'of the breathing rate distribution.'
    )
    doc.add_paragraph(
        'For cardiac rate, the mean bias was -7.9 BPM (CAP underestimates) and '
        'the limits of agreement were wide: [-44.8, +28.9] BPM. A 73 BPM '
        'spread between the upper and lower limits is not clinically useful. '
        'The scatter plot shows a dense core of epochs near the zero line, but '
        'with a substantial tail of epochs where the CAP estimate deviates by '
        '20-80 BPM. These large errors are concentrated in sessions S6N1 and '
        'S6N2 (Table 1), where the cardiac signal appears to be degraded -- '
        'S6N2 alone has a median MAE of 71.8 BPM, consistent with either sensor '
        'contact failure or an anatomical variant that prevents BCG transmission '
        'to the temple sensors.'
    )

    add_figure(
        doc,
        FIGURES / 'phase5_bland_altman.png',
        'Figure 5. Bland-Altman agreement: multi-channel fused + Viterbi '
        'smoothed CAP rate vs PSG ground truth. Left: respiratory (bias '
        '+0.2 br/min, LoA [-5.0, +5.5]) with visible proportional bias. '
        'Right: cardiac (bias -7.9 BPM, LoA [-44.8, +28.9]) with wide spread '
        'driven by outlier sessions.',
        width=6.0
    )

    # --- Per-stage ---
    doc.add_heading('Accuracy by Sleep Stage', level=3)
    doc.add_paragraph(
        'Respiratory MAE was relatively stable across sleep stages: 1.5 br/min '
        'in N2, 1.6 br/min in Wake, N1, and N3, rising to 2.5 br/min in REM '
        '(Figure 6, left). The REM degradation likely reflects increased '
        'breathing irregularity during phasic REM, which broadens the spectral '
        'peak and reduces the accuracy of single-frequency estimation. The REM '
        'sample was small (n=218) and this estimate should be treated with '
        'caution.'
    )
    doc.add_paragraph(
        'Cardiac MAE varied more across stages: best during N1 (5.5 BPM), '
        'followed by N3 (6.1 BPM), N2 (6.8 BPM), Wake (7.1 BPM), and REM '
        '(7.9 BPM) (Figure 6, right). The N1 advantage is unexpected -- light '
        'sleep might be assumed to have more variability -- but may reflect the '
        'fact that N1 has the most regular heart rate in our cohort, making the '
        'spectral peak sharper and easier to track. Wake and REM are worst, '
        'likely due to autonomic variability (Wake) and phasic cardiac '
        'irregularity (REM).'
    )

    add_figure(
        doc,
        FIGURES / 'phase5_per_stage_mae.png',
        'Figure 6. Median MAE by sleep stage for the best pipeline. Left: '
        'respiratory -- stable at 1.5-1.6 br/min except REM (2.5). Right: '
        'cardiac -- N1 best (5.5 BPM), Wake and REM worst (7.1, 7.9 BPM). '
        'Sample sizes shown above each bar.',
        width=6.2
    )

    # --- Per-session table ---
    doc.add_heading('Per-Session Results', level=3)

    csv_path = FIGURES / 'phase5_per_session_summary.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)

        for band, unit in [('resp', 'br/min'), ('card', 'BPM')]:
            sub = df[df.band == band][['session', 'MAE', 'RMSE', 'bias', 'r', 'n_epochs']].copy()
            sub['MAE'] = sub['MAE'].round(1)
            sub['RMSE'] = sub['RMSE'].round(1)
            sub['bias'] = sub['bias'].round(1)
            sub['r'] = sub['r'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else '--')
            sub.columns = ['Session', f'MAE ({unit})', f'RMSE ({unit})', f'Bias ({unit})', 'r', 'Epochs']

            label = 'Respiratory' if band == 'resp' else 'Cardiac'
            tbl_num = '1a' if band == 'resp' else '1b'
            add_table(
                doc, sub,
                f'Table {tbl_num}. {label} rate accuracy per session '
                f'(confidence-weighted fusion + Viterbi, no k-scaling). '
                f'r = Pearson correlation between CAP and PSG per-window rates.'
            )

    doc.add_paragraph(
        'Table 1 reveals two problems that the pooled statistics obscure. '
        'First, the per-window correlation (r) between CAP and PSG rates is '
        'near zero or negative in almost every session, for both bands. A '
        'median MAE of 1.5 br/min or 6.6 BPM sounds reasonable in isolation, '
        'but the near-zero correlation means the pipeline tracks the average '
        'rate over the night without following the window-to-window fluctuations. '
        'The Viterbi smoother contributes to this: by penalizing rapid changes, '
        'it biases the output toward a running mean and suppresses the very '
        'variations that would produce a positive correlation with ground truth.'
    )
    doc.add_paragraph(
        'Second, session S6N2 is a catastrophic failure for cardiac rate '
        '(MAE 71.8 BPM, bias -61.4 BPM). This session also had poor results '
        'in the k-scaled analysis. Excluding S6N2, the cardiac median MAE '
        'across the remaining 11 sessions is 5.8 BPM -- still higher than the '
        'k-scaled result (4.19 BPM) but not unreasonable. The respiratory '
        'results are more consistent: worst-case MAE is 2.6 br/min (S3N2), and '
        'the best sessions (S5N1, S5N2) achieve 0.6 br/min.'
    )

    # --- Pipeline comparison ---
    doc.add_heading('Pipeline Comparison', level=3)
    doc.add_paragraph(
        'Figure 7 summarizes all pipeline variants. For respiratory rate, '
        'every variant produces the same 1.5 br/min -- the choice of fusion '
        'strategy and smoothing is irrelevant. For cardiac rate, the ordering is '
        'clear: confidence-weighted fusion + Viterbi (6.6 BPM) is best, '
        'followed by agreement-filtered + Viterbi (7.8 BPM), then best-fixed + '
        'Viterbi (8.6 BPM). The raw (unsmoothed) variants are 31-35% worse in '
        'every case.'
    )

    add_figure(
        doc,
        FIGURES / 'phase5_pipeline_comparison.png',
        'Figure 7. Full pipeline comparison: all fusion strategies with and '
        'without Viterbi smoothing. Respiratory rate is invariant to pipeline '
        'choice. Cardiac rate benefits from both fusion (19% improvement) and '
        'smoothing (35% improvement).',
        width=6.2
    )


# ==============================================================================
# DISCUSSION
# ==============================================================================

def write_discussion(doc: Document):
    doc.add_heading('Discussion', level=1)
    doc.add_heading('Can the K-Factor Be Eliminated?', level=2)

    doc.add_paragraph(
        'The answer depends on which rate. For respiratory rate, yes: the '
        'spectral method achieves 1.5 br/min MAE without any PSG calibration, '
        'outperforming the k-scaled peak counting method (2.20 br/min) from the '
        'previous analysis. This is because the spectral approach estimates the '
        'dominant frequency directly rather than counting individual breath '
        'events, so the overcounting problem that k-factor correction was '
        'designed to fix does not arise. There is no reason to use k-scaling '
        'for respiratory rate estimation from these sensors.'
    )
    doc.add_paragraph(
        'For cardiac rate, no: the best PSG-free pipeline (confidence-weighted '
        'fusion + Viterbi) achieves 6.6 BPM MAE, compared to 4.19 BPM with '
        'k-scaling -- 57% worse. The BCG waveform produces multiple mechanical '
        'impulses per heartbeat, and no method we tested fully resolves the '
        'fundamental rate without calibration. CWT ridge tracking comes closest '
        '(11.6 BPM raw, 6.6 BPM after fusion and smoothing), but the remaining '
        'error is not negligible. The k-factor approach, despite requiring an '
        'initial PSG reference, remains the more accurate option for cardiac '
        'rate.'
    )

    doc.add_heading('Limitations of Temporal Smoothing', level=2)
    doc.add_paragraph(
        'The Viterbi smoother reduced cardiac MAE by 35%, which appears to '
        'be a major improvement. But this improvement comes at a cost that the '
        'MAE statistic alone does not reveal. The smoother works by penalizing '
        'epoch-to-epoch rate changes, which suppresses both noise and genuine '
        'physiological variation. The result is a rate trace that stays close '
        'to the session mean but fails to track real fluctuations -- hence '
        'the near-zero per-window correlations in Table 1. A smoothed estimate '
        'with low MAE and zero correlation is not the same as an accurate '
        'estimate; it is a biased estimate that happens to be close to the '
        'mean in the median case.'
    )
    doc.add_paragraph(
        'For applications that require only an average rate over the night '
        '(e.g., screening for sleep-disordered breathing), this may be '
        'acceptable. For applications that require tracking rate dynamics '
        '(e.g., autonomic staging, apnea detection), the smoothed pipeline is '
        'not suitable in its current form. A more sophisticated approach -- '
        'adaptive smoothing that varies its constraint strength based on '
        'local signal quality -- might preserve dynamics in high-SNR windows '
        'while still stabilizing noisy ones.'
    )

    doc.add_heading('Channel Fusion: Limited Returns', level=2)
    doc.add_paragraph(
        'Multi-channel fusion provided only modest cardiac improvement (19%) '
        'and no respiratory improvement at all. The oracle analysis shows that '
        'per-window optimal channel selection could achieve 4.5 BPM cardiac MAE '
        '-- close to the k-scaled result -- but our quality-based selector '
        'captures only a fraction of this. The quality score (SNR, spectral '
        'concentration, ACF prominence) is evidently not a good predictor of '
        'which channel will be closest to ground truth in any given window. '
        'A learned channel selector, trained on the relationship between quality '
        'features and per-channel error, might close this gap, but would itself '
        'require labeled data for training.'
    )
    doc.add_paragraph(
        'The respiratory result is informative in a different way: the fact '
        'that all channels produce identical accuracy means the respiratory '
        'signal is spatially homogeneous across the sensor array. There is no '
        'preferred sensor position for breathing rate estimation from the sleep '
        'mask, and any single sensor is sufficient.'
    )

    doc.add_heading('Comparison with K-Scaled Pipeline', level=2)

    # Comparison table
    comp_data = {
        'Pipeline': [
            'Peaks/k (calibrated)',
            'Spectral (no calibration)',
            'Fused + Viterbi (no calibration)',
        ],
        'Resp MAE (br/min)': ['2.20', '1.5', '1.5'],
        'Card MAE (BPM)': ['4.19', '15.5', '6.6'],
        'Requires PSG': ['Yes', 'No', 'No'],
        'Tracks dynamics': ['Partially', 'No', 'No'],
    }
    comp_df = pd.DataFrame(comp_data)
    add_table(
        doc, comp_df,
        'Table 2. Comparison of rate estimation pipelines. The PSG-free '
        'spectral method is superior for respiratory rate. The k-scaled '
        'pipeline remains superior for cardiac rate.'
    )

    doc.add_paragraph(
        'The k-factor approach and the multi-channel fusion approach are not '
        'competitors -- they address different deployment scenarios. In a '
        'laboratory setting where PSG is available for one calibration session, '
        'k-scaling provides better cardiac accuracy and preserves some '
        'window-level dynamics. In a home or clinical screening setting where '
        'PSG is not available, the multi-channel pipeline provides usable '
        'respiratory rate and approximate cardiac rate, with the caveat that '
        'the cardiac estimate reflects the session average more than '
        'moment-to-moment variation.'
    )

    doc.add_heading('Session-Level Failures', level=2)
    doc.add_paragraph(
        'Session S6N2 produced a cardiac MAE of 71.8 BPM, effectively a '
        'complete failure. This session was also an outlier in the k-scaled '
        'analysis (k = 0.79, indicating undercounting rather than the typical '
        'overcounting). The most likely explanation is poor sensor-skin contact '
        'on the cardiac-dominant sensor, degrading the BCG signal below the '
        'noise floor. Session S6N1 from the same subject was also poor '
        '(11.4 BPM), suggesting a subject-specific anatomical factor -- '
        'possibly thicker temporal bone or lower-amplitude cardiac pulsations '
        'at the temple site.'
    )
    doc.add_paragraph(
        'Excluding subject S6 entirely, the cardiac MAE for the remaining 10 '
        'sessions is 5.6 BPM, which is clinically borderline. But excluding '
        'the worst subject is exactly the kind of post-hoc analysis that '
        'inflates apparent performance. Any deployment of this technology '
        'must anticipate that some fraction of users will produce unusable '
        'cardiac data, and the system must detect and report this failure '
        'rather than silently returning inaccurate rates.'
    )


# ==============================================================================
# Main
# ==============================================================================

def main():
    doc = Document()
    setup_styles(doc)

    for section in doc.sections:
        section.top_margin = Cm(2.54)
        section.bottom_margin = Cm(2.54)
        section.left_margin = Cm(2.54)
        section.right_margin = Cm(2.54)

    # Title
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(
        'Multi-Channel Rate Estimation from Capacitive Temple Sensors\n'
        'Without PSG Calibration'
    )
    run.font.size = Pt(14)
    run.font.bold = True
    run.font.name = 'Times New Roman'

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run('Supplementary section for integration into main paper')
    run.font.size = Pt(10)
    run.font.italic = True
    doc.add_paragraph()

    write_methods(doc)
    write_results(doc)
    write_discussion(doc)

    doc.save(str(OUT))
    print(f'Saved to {OUT}')
    print(f'  Sections: Methods, Results (7 figures, 3 tables), Discussion')


if __name__ == '__main__':
    main()
