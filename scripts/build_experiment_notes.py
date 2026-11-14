"""Build an editable Markdown report and a paginated PDF from saved evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, PageBreak, KeepTogether)
from reportlab.graphics.shapes import Drawing, Line, Circle, String

NAMES={'manual':'Hand-weighted rules','rule_lr':'Rule-feature LR','relation_only':'Relation prior only',
       'rule_relation':'Rules + relation LR','context_only':'Context + relation LR','context_lr':'Rules + context + relation LR'}
ORDER=list(NAMES)
def pct(x): return '-' if x is None else f'{100*x:.2f}'


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--results',type=Path,default=Path('results/20261114'))
    parser.add_argument('--out',type=Path,default=Path('report')); args=parser.parse_args()
    root=args.results; out=args.out; out.mkdir(parents=True,exist_ok=True)
    read=lambda name:json.loads((root/name).read_text())
    r=read('results.json'); audits=read('diagnostic_audits.json'); dev=read('dev_audit.json')
    # Tables are generated; the accompanying prose was reviewed for this release.
    # Fail rather than pair newly changed results with an old narrative.
    metric_payload={v:{k:x for k,x in values.items() if k not in ('folds','development_experiment_seconds')} for v,values in r.items()}
    actual_digest=hashlib.sha256(json.dumps(metric_payload,sort_keys=True).encode()).hexdigest()
    reference=Path(__file__).resolve().parents[1]/'configs/report_reference.json'
    if actual_digest!=json.loads(reference.read_text())['metrics_sha256']:
        raise ValueError('Results differ from the reviewed report. Revise the narrative before changing report_reference.json.')
    cis=read('oof_paired_bootstrap.json'); timing=read('timings.json'); selection=read('selection.json')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Title2',fontName='Helvetica-Bold',fontSize=23,leading=27,textColor=colors.HexColor('#163547'),spaceAfter=15))
    styles.add(ParagraphStyle(name='Body2',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=9))
    styles.add(ParagraphStyle(name='Small2',fontName='Helvetica',fontSize=8.2,leading=11,spaceAfter=7))
    styles['Heading1'].fontSize=16; styles['Heading1'].leading=20; styles['Heading1'].textColor=colors.HexColor('#163547')
    story=[]; md=[]
    def para(text,small=False):
        story.append(Paragraph(escape(text),styles['Small2' if small else 'Body2'])); md.append(text+'\n')
    def head(text):
        story.append(Paragraph(escape(text),styles['Heading1'])); md.append('## '+text+'\n')
    def table(headers,rows,widths=None):
        data=[[Paragraph(escape(str(x)),styles['Small2']) for x in row] for row in [headers]+rows]
        t=Table(data,colWidths=widths or [468/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eff2')),('VALIGN',(0,0),(-1,-1),'TOP'),
                               ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#64818e')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7f8')]),
                               ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
        story.extend([t,Spacer(1,10)])
        md.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |')
        md.extend('| '+' | '.join(map(str,row))+' |' for row in rows); md.append('')
    def page(): story.append(PageBreak()); md.append('\n---\n')
    def metric_rows(split,policy,variants=ORDER):
        return [[NAMES[v],pct(r[v][split][policy]['overall']['clean_retention']),pct(r[v][split][policy]['overall']['noise_recall']),pct(r[v][split][policy]['overall']['retained_purity']),f"{r[v][split][policy]['overall']['roc_auc']:.3f}"] for v in variants]

    story.append(Paragraph('Retaining Useful Supervision<br/>in Lightweight Relation Denoising',styles['Title2']))
    md.append('# Retaining Useful Supervision in Lightweight Relation Denoising\n')
    para('Zijun Wu | Experimental report | 14 November 2026')
    para('The project began in an NYU NLP course with Zijun Wu, Tianchen Xie, and Tianluo Zhang. The implementation and evaluation documented here were prepared with coding-agent assistance; contributions are recorded in docs/PROVENANCE.md.',True)
    head('Abstract')
    para('A filter can achieve high noise recall by discarding most of its useful supervision. We revisit a hand-written linguistic filter and compare six lightweight models under an explicit clean-retention objective. A sparse L2 logistic model combines the original rules with entity-masked local words and bigrams, candidate-relation indicators, and relation-feature interactions. Five-fold nested sentence-grouped evaluation on 4,241 labeled development examples yields 89.93% clean retention and 31.84% noise recall, compared with 90.94% and 19.58% for a rule-feature logistic baseline. On 3,966 diagnostic test examples without development-sentence overlap, the new model retains 92.37% of clean examples and removes 30.44% of noise. A separate, smaller noise-focused diagnostic does not show the same improvement. The evidence supports a better measured retention-quality trade-off, not a neural benchmark win or a downstream relation-extraction claim.')
    head('1. Research question and rationale')
    para('The task is to decide whether a supplied relation label is supported by a sentence mentioning two entities. An incorrect NA label is also noise: NA does not automatically mean an example is useless. We therefore evaluate both NA and non-NA examples and select separate operating thresholds for these groups.')
    para('The previous recall-first objective encouraged removal of roughly 90% of labeled noise even when correct examples were also removed. This release separates two questions: does a model rank label validity better, and which operating point preserves useful data? Lowering the removal rate alone is not evidence of a better model.')
    head('2. Model progression')
    para('Hand-written rules score entity distance, intervening-text length, relation keywords, and entity order, and penalize uncertainty, list-like text, and overlapping entities. Rule-feature logistic regression learns how to combine eight such components and an NA indicator. The new sparse model additionally conditions on the candidate relation, learns relation-specific rule weights, and reads entity-masked local unigrams and bigrams. Its coefficients remain inspectable; no pretrained encoder or GPU is required.')
    page()
    head('3. Data and evaluation protocol')
    para('Data: the authors\' ARNOR 2.0 corpus. This release is a supervised validity-filter experiment using labeled development data; it is not a reimplementation of ARNOR\'s distant-supervision training protocol. All files are identified by SHA-256 in configs/data_manifest.json. Raw news articles are not redistributed.')
    table(['Split','Raw mentions','Valid labeled, unique','Sentence groups','After dev-text exclusion'],[
        ['Development',str(dev['input_records']),str(dev['eligible_records']),str(dev['sentence_groups']),'-'],
        ['Test',str(audits['test']['input_records']),str(audits['test']['eligible_records']),str(audits['test']['sentence_groups']),str(audits['test']['without_dev_text'])],
        ['Noise-focused test',str(audits['test_noise']['input_records']),str(audits['test_noise']['eligible_records']),str(audits['test_noise']['sentence_groups']),str(audits['test_noise']['without_dev_text'])]], [108,68,106,80,106])
    para('Entity spans must resolve uniquely and match the text. We quarantine 326 development, 425 test, and 200 noise-focused mentions with ambiguous spans. We remove 9 duplicate test annotations and 26 duplicate noise-focused annotations; 4 unknown and 2 conflicting noise-focused labels are excluded. Counts are disclosed rather than silently treated as clean. Production inference emits an explicit abstention for uncertain spans.')
    para('Primary evaluation: five outer folds grouped by normalized sentence text; all entity mentions of a sentence stay together. Within each outer training set, three inner grouped folds select C from {0.1, 1, 10} by minimum log loss, then select thresholds from inner predictions. Vectorizer vocabulary and scalers fit only on each training partition. L2 logistic regression uses liblinear; the rule-only comparator uses scaled rule features, while sparse variants use maximum-absolute scaling. All models use the same folds, search budget, and decision objectives.')
    para('Retain90 maximizes noise recall subject to at least 90% clean retention separately for NA/non-NA on inner predictions. Retain95 is a conservative alternative. Recall90 maximizes clean retention subject to 90% noise recall. Thresholds use exact tied-score boundaries. These are empirical selection constraints, not guarantees that every outer fold or new population will meet the target.')
    para('The protocol was saved before the run. All final models, thresholds, and the development-based promotion decision were frozen before test labels were read in this run. The context model passed the predeclared development gate. However, official development and test data were examined in earlier project iterations. Neither this test nor the development OOF estimate is advertised as a pristine external evaluation. Model selection based on OOF evidence also creates selection uncertainty.')
    head('Metric definitions')
    para('Noise is the positive class. TP = noisy example removed; FP = clean example removed; FN = noisy example retained; TN = clean example retained. Noise recall = TP/(TP+FN); clean retention = TN/(TN+FP); retained-set purity = TN/(TN+FN); noise precision = TP/(TP+FP). F1 refers only to noise detection. Undefined denominators remain null. These metrics cannot be substituted for relation-classification F1.')
    page()
    head('4. Primary development results')
    para('All values below except AUC are percentages. These are pooled outer-fold predictions, not fitted training performance. Keep-all would retain 100% of clean examples, remove 0% of noise, and have 75.19% purity on this development set.')
    table(['Model','Clean retained','Noise removed','Kept purity','ROC AUC'],metric_rows('oof','retain90'),[180,72,72,72,72])
    para('At approximately 90% clean retention, the new context model removes 12.26 percentage points more noise than rule-feature LR. A paired bootstrap over sentence groups gives a conditional 95% interval of +8.72 to +15.94 points. Clean retention differs by -1.00 points (interval -3.01 to +1.14). The interval conditions on fitted OOF predictions; it does not include retraining or design-selection variability.')
    para('The raw hand-weighted score keeps 97.99% of clean examples and removes only 2.00% of noise. Score ties, especially for NA, prevent a deterministic threshold from spending the full allowed rejection budget. This row is not matched to the others at exactly the same retention; its threshold-independent AUC is also reported.')
    head('5. Main diagnostic test')
    para('This table excludes the 84 eligible mentions whose normalized sentence also occurs in development. It contains 2,987 clean and 979 noisy examples (3,966 total). Thresholds and model choices are fixed from development.')
    table(['Model','Clean retained','Noise removed','Kept purity','ROC AUC'],metric_rows('test_no_dev_text','retain90'),[180,72,72,72,72])
    para('The selected model retains 2,759 clean examples and removes 298 noisy examples; rule-feature LR retains 2,699 and removes 214. Thus, on this diagnostic set the new model preserves 60 additional clean examples while removing 84 additional noisy examples. This is an improvement in both measured outcomes under the frozen policies, rather than a direct numerical comparison with the historical 14.30% retention run, which used a different objective and sample handling.')
    page()
    head('6. What the ablations establish')
    para('Learning rule weights improves AUC over a hand-weighted total. Allowing relation-specific coefficients further raises development AUC from 0.643 to 0.681. Adding masked lexical context raises it to 0.745. Compared with rules plus relation features, the context model increases OOF noise recall by 3.99 points with a sentence-bootstrap interval of +0.91 to +7.26, at similar clean retention. Candidate-relation priors alone do not explain the full improvement.')
    para('Removing the hand-designed rule features from the context model produces nearly identical OOF performance: 31.75% versus 31.84% noise recall. The paired difference interval includes zero. The evidence therefore does not establish that the original rules add a substantial gain once local context is available. The rules remain valuable as a transparent starting hypothesis and an interpretable baseline, rather than being credited with an unsupported causal contribution.')
    head('7. Operating points and the retention problem')
    table(['Method / objective','Dev clean retained','Dev noise removed','Test clean retained','Test noise removed'],[
        [NAMES[v]+' / '+policy,pct(r[v]['oof'][policy]['overall']['clean_retention']),pct(r[v]['oof'][policy]['overall']['noise_recall']),pct(r[v]['test_no_dev_text'][policy]['overall']['clean_retention']),pct(r[v]['test_no_dev_text'][policy]['overall']['noise_recall'])]
        for v,policy in [('rule_lr','recall90'),('context_lr','recall90'),('rule_lr','retain90'),('context_lr','retain90'),('context_lr','retain95')]], [180,72,72,72,72])
    para('Changing the objective explains part of the large increase in retained clean data. Improved discrimination explains another part: under the same Recall90 selection rule, the context model retains 38.13% of clean examples on the main diagnostic versus 26.55% for rule-feature LR, while removing 91.52% versus 88.97% of noise. Under Retain95, the new model preserves 96.02% of clean data but removes only 16.34% of noise. No single threshold simultaneously solves both objectives.')
    drawing=Drawing(468,200); left=45; bottom=30; width=390; height=145
    for value in [0,25,50,75,100]:
        x=left+width*value/100; y=bottom+height*value/100
        drawing.add(Line(x,bottom,x,bottom+height,strokeColor=colors.HexColor('#e1e7ea')))
        drawing.add(Line(left,y,left+width,y,strokeColor=colors.HexColor('#e1e7ea')))
        drawing.add(String(x-6,15,str(value),fontSize=8)); drawing.add(String(15,y-3,str(value),fontSize=8))
    for v,col in [('rule_lr','#bb633e'),('context_lr','#197a85')]:
        points=[]
        for policy in ['recall90','retain90','retain95']:
            m=r[v]['test_no_dev_text'][policy]['overall']; x=left+width*m['clean_retention']; y=bottom+height*m['noise_recall']; points.append((x,y)); drawing.add(Circle(x,y,3,fillColor=colors.HexColor(col),strokeColor=colors.HexColor(col)))
        for a,b in zip(points,points[1:]):drawing.add(Line(*a,*b,strokeColor=colors.HexColor(col),strokeWidth=1.2))
    drawing.add(String(110,0,'Clean examples retained (%)',fontSize=9));drawing.add(String(45,184,'Noise removed (%)',fontSize=9))
    drawing.add(String(240,184,'Rule LR',fontSize=9,fillColor=colors.HexColor('#bb633e')));drawing.add(String(320,184,'Context LR',fontSize=9,fillColor=colors.HexColor('#197a85')))
    story.append(drawing);md.append('Plot data: policy_tradeoffs.csv. Lines connect three development-selected operating points; they are not a complete frontier.\n')
    para('Three development-selected operating points on the main diagnostic; lines are visual guides, not interpolated deployment policies.',True)
    page()
    head('8. Robustness and failure analysis')
    para('On the 2,735 main test examples with neither a development sentence nor a development entity pair, the context model retains 94.04% of clean examples and removes 32.87% of noise. Rule-feature LR retains 90.40% and removes 18.22%. This sensitivity analysis reduces one source of overlap; it does not establish generalization to another corpus or resolve prior human exposure to the test set.')
    table(['Main test subgroup','Model','Clean retained','Noise removed'],[
        [g,NAMES[v],pct(r[v]['test_no_dev_text']['retain90'][g]['clean_retention']),pct(r[v]['test_no_dev_text']['retain90'][g]['noise_recall'])]
        for g in ['NA','non-NA'] for v in ['rule_lr','context_lr']], [90,210,84,84])
    para('The small noise-focused diagnostic has a different composition: 183 of its 297 eligible non-overlapping examples are labeled noise, compared with 979 of 3,966 in the main diagnostic. It also loses many examples under strict span validation. We report it as a separate stress case, not as a directly comparable ARNOR benchmark reproduction.')
    table(['Noise-focused method','Clean retained','Noise removed','Noise F1','ROC AUC'],[
        [NAMES[v],pct(r[v]['test_noise_no_dev_text']['retain90']['overall']['clean_retention']),pct(r[v]['test_noise_no_dev_text']['retain90']['overall']['noise_recall']),pct(r[v]['test_noise_no_dev_text']['retain90']['overall']['noise_f1']),f"{r[v]['test_noise_no_dev_text']['retain90']['overall']['roc_auc']:.3f}"] for v in ['manual','rule_lr','context_only','context_lr']], [180,72,72,72,72])
    para('The context model is weaker than rule-feature LR on this stress case: 33.88% versus 41.53% noise recall. Its AUC is also lower (0.664 versus 0.788). A plausible explanation is that the lexical features fit the development distribution better than this selected noise set, but this experiment does not isolate the cause. No architecture or threshold was retuned after seeing these outcomes.')
    head('9. CPU footprint and engineering')
    table(['Model','Examples','Median seconds','JSON bytes'],[[NAMES[v],str(timing[v]['n']),f"{timing[v]['median_seconds']:.3f}",f"{timing[v]['model_json_bytes']:,}"] for v in ['rule_lr','context_lr']],[180,72,108,108])
    para('Timing includes feature extraction, portable JSON scoring, and decisions on 4,050 validated examples. It excludes file IO. There was one warm-up and four measured repeats, with one numerical-library thread requested. These local CPU measurements do not support a speedup ratio against BERT or ARNOR. Model coefficients are plain JSON rather than executable pickle. Invalid spans generate abstentions in the inference CLI.')
    page()
    head('10. Interpretation and next experiments')
    para('The experiments support a bounded conclusion: relation-conditioned local context improves this lightweight validity filter on the development evaluation and the main diagnostic. The progression from hand-written linguistic hypotheses to learned coefficients and local lexical evidence separates representation quality from operating-point selection. The context-only ablation suggests that lexical features explain most of the additional gain; the noise-focused diagnostic shows that the improvement is distribution-dependent. This release establishes a reproducible supervised sparse baseline, not state-of-the-art performance.')
    para('Limitations: the labeled development set is small, many rare relations have very few examples, ambiguous spans are excluded, outer-fold retention fluctuates, and the corpus has been inspected before. Confidence intervals are conditional, the probabilities are not externally calibrated, and choosing a model from development OOF results adds selection uncertainty. The context-only ablation weakens any claim that handcrafted rules are essential. The specialized noise set reveals distribution sensitivity.')
    para('The next validation should be fixed before obtaining new results: independently annotate a fresh sentence-grouped holdout; examine rare-relation and domain shift; compare downstream relation extractors trained on all data, filtered data, and matched-size random subsets; and evaluate abstention/review and probability calibration. No downstream gains or neural comparisons are claimed in this release.')
    head('11. Reproduction and provenance')
    para('Install the package with the locked numerical dependencies. Download the official corpus using scripts/download_data.py, which verifies all four file hashes. Run python -m relden.experiment with a new results directory. Run scripts/verify_release.py to independently recalculate the saved metrics, verify sentence-group isolation, and check the frozen model hashes. The tests cover label leakage, entity masking, tied-score thresholds, export parity, grouped folds, and handling of ambiguous/unknown/conflicting records.')
    para('The project progressed from hand-designed rules to learned weights and context-sensitive filtering. Team contributions and coding-agent assistance are documented in docs/PROVENANCE.md. The final repository consolidates the code and results into one release commit, with earlier history backed up separately. Only results produced by the current evaluation scripts are used in this report.')
    head('References')
    para('[1] Jia, W., Dai, D., Xiao, X., and Wu, H. (2019). ARNOR: Attention Regularization based Noise Reduction for Distant Supervision Relation Classification. ACL, 1399-1408. https://aclanthology.org/P19-1135/',True)
    para('[2] ARNOR authors\' official dataset and version-2 documentation. https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2019-ARNOR',True)
    para('[3] Scikit-learn documentation: grouped cross-validation and decision-threshold tuning. https://scikit-learn.org/stable/modules/cross_validation.html ; https://scikit-learn.org/stable/modules/classification_threshold.html',True)
    para('[4] Project source history: https://github.com/LewisZZZ-Zhang/NLP_Final_Project ; consolidated release: https://github.com/JonathanWu04/interpretable-relation-denoising',True)
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#c4d0d5'));canvas.line(54,43,558,43)
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#58707c'))
        canvas.drawString(54,30,'Relation Denoising | Experimental report | 14 Nov 2026')
        canvas.drawRightString(558,30,str(doc.page))
    SimpleDocTemplate(str(out/'experiment_notes.pdf'),pagesize=(612,792),leftMargin=72,rightMargin=72,topMargin=48,bottomMargin=60,
                      title='Retaining Useful Supervision in Lightweight Relation Denoising',author='Zijun Wu').build(story,onFirstPage=footer,onLaterPages=footer)
    (out/'technical_report.md').write_text('\n'.join(md))
    with (out/'policy_tradeoffs.csv').open('w') as f:
        writer=csv.writer(f,lineterminator="\n");writer.writerow(['model','split','policy','clean_retention','noise_recall','retained_purity'])
        for v in ORDER:
            for split in ['oof','test_no_dev_text','test_noise_no_dev_text']:
                for policy in ['retain90','retain95','recall90']:
                    m=r[v][split][policy]['overall'];writer.writerow([v,split,policy,m['clean_retention'],m['noise_recall'],m['retained_purity']])
    print('Built',out/'experiment_notes.pdf')

if __name__=='__main__': main()
