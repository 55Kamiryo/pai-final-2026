"""Create the submission MP4, PDF and source ZIP for camera stop/recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
FPS = 20
BG, WHITE, MUTED, TEAL, BLUE = '#101D2A', '#F2F6F8', '#A8BAC8', '#59D4BB', '#8BB9F6'


def decode(path, width, height):
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'])
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)


def outcome(row):
    return '達成' if row['success'] else '未達成'


def outcome_sentence(row):
    seconds = row['simulated_seconds']
    return f'{seconds:.2f}秒でタスク達成' if row['success'] else f'{seconds:.2f}秒で打切り、未達成'


def experiment_facts(input_root, data, rows):
    """Verify claims against saved inputs/actions before authoring a report."""
    for key in ['task', 'model_sha256', 'control_hz', 'max_steps', 'settings']:
        if any(d[key] != data['normal'][key] for d in data.values()):
            raise ValueError(f'Comparison conditions differ: {key}')
    for key in ['seed', 'initial_state_index']:
        if any(r[key] != rows['normal'][key] for r in rows.values()):
            raise ValueError(f'Comparison conditions differ: {key}')
    if data['normal']['control_hz'] != FPS:
        raise ValueError('This video layout requires 20 Hz recordings')
    fault = data['guarded']['camera_fault']
    if fault != data['unprotected']['camera_fault']:
        raise ValueError('Camera fault conditions differ')
    events = data['guarded']['camera_recovery_events']
    if len(events) != 2 or [e['event'] for e in events] != ['pause', 'resume_with_new_plan']:
        raise ValueError('Expected one complete pause/recovery event pair')
    with np.load(input_root/'guarded/wrist_inputs.npz', allow_pickle=False) as capture:
        paused = capture['paused'].copy()
        predicted = capture['model_prediction'].copy()
    with np.load(input_root/'guarded'/rows['guarded']['trajectory'], allow_pickle=False) as trajectory:
        actions = trajectory['actions'].copy()
    if len(actions) != rows['guarded']['steps'] or len(paused) != len(actions):
        raise ValueError('Guarded action/input lengths disagree')
    if not np.array_equal(np.flatnonzero(paused), np.arange(events[0]['step'], events[1]['step'])):
        raise ValueError('Recorded pause interval disagrees with events')
    if np.any(actions[paused, :6] != 0) or np.any(predicted[paused]):
        raise ValueError('Pause did not hold arm commands or suppress inference')
    before = max(events[0]['step'] - 1, 0)
    if np.any(actions[paused, 6] != actions[before, 6]):
        raise ValueError('Gripper command changed during the pause')
    for row in rows.values():
        if not np.isclose(row['simulated_seconds'], row['steps']/FPS):
            raise ValueError('Recorded duration disagrees with step count')
    return {
        'pause': events[0]['step'], 'resume': events[1]['step'],
        'start': fault['start_step'], 'end': fault['end_step_exclusive'],
        'paused_steps': int(paused.sum()), 'seed': rows['normal']['seed'],
        'initial_state_index': rows['normal']['initial_state_index'],
    }


def recovery_sentence(facts, row):
    return (
        f"step {facts['pause']}（{facts['pause']/FPS:.2f}秒）で停止、"
        f"step {facts['end']}（{facts['end']/FPS:.2f}秒）で画像復旧、"
        f"step {facts['resume']}（{facts['resume']/FPS:.2f}秒）で再計画・再開し、"
        f"step {row['steps']}（{row['simulated_seconds']:.2f}秒）で"
        + ('達成した。' if row['success'] else '打ち切り、未達成だった。')
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT/'results/camera-recovery')
    p.add_argument('--output', type=Path, default=ROOT/'output')
    p.add_argument('--account', required=True, help='OmniCampus account name or ID for submission')
    p.add_argument('--font', type=Path, default=Path('/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'))
    p.add_argument('--pdf-font', type=Path, default=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
    p.add_argument('--pdf-only', action='store_true')
    args = p.parse_args()
    if not args.account or any(c in args.account for c in '/\\\n\r'):
        raise ValueError('Invalid account name')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    data = {name: json.loads((args.input/name/'results.json').read_text()) for name in ['normal', 'unprotected', 'guarded']}
    rows = {name: value['episodes'][0] for name, value in data.items()}
    facts = experiment_facts(args.input, data, rows)
    start, end = facts['start']/FPS, facts['end']/FPS
    if args.pdf_only:
        make_pdf(args, data, rows, facts)
        return
    fonts = {n: ImageFont.truetype(str(args.font), n) for n in [18, 20, 22, 24, 26, 28, 30, 36, 42, 48]}

    def text(draw, xy, value, size=28, color=WHITE):
        draw.text(xy, value, font=fonts[size], fill=color)

    def base(section):
        im = Image.new('RGB', (1280,720), BG)
        d = ImageDraw.Draw(im)
        text(d, (48,24), 'PAI 2026 / '+section, 20, TEAL)
        text(d, (1000,24), args.account, 20, MUTED)
        d.line((48,62,1232,62), fill='#304657')
        return im

    preview_dir = out/'camera-submission-preview'
    preview_dir.mkdir(exist_ok=True)
    comparison = out/'camera-recovery-preview-with-inputs.mp4'
    subprocess.run([
        sys.executable, str(ROOT/'experiments/build_camera_preview.py'),
        '--input', str(args.input), '--output', str(comparison), '--font', str(args.font),
    ], check=True)
    compare = decode(comparison,1280,720)
    middle = (facts['start'] + facts['end'])//2
    for index, name in [(middle,'camera_blackout.png'),(facts['resume'],'camera_recovered.png'),(rows['guarded']['steps'],'camera_completed.png')]:
        Image.fromarray(compare[min(index,len(compare)-1)]).save(out/name)
    normal = decode(args.input/'normal'/rows['normal']['video'],512,512)
    with np.load(args.input/'guarded/wrist_inputs.npz', allow_pickle=False) as z:
        wrist = z['images'].copy()

    intro = base('CAMERA LOSS & RECOVERY')
    d = ImageDraw.Draw(intro)
    text(d,(56,115),'画像が消えたら止まり、',48)
    text(d,(56,190),'戻ったら片付けを再開する',48)
    text(d,(58,300),'SmolVLAによる器の移動 ＋ カメラ入力の異常監視',30,TEAL)
    text(d,(58,398),'AIの推論と、今回追加した停止・復旧処理を組み合わせます。',26)
    text(d,(58,498),'比較：同じ画像欠損に対する、処理あり・なしの動作',26)
    text(d,(58,596),'1タスク・1配置・各条件1回の実証 / 追加学習なし',24,MUTED)

    ai = base('TASK & AI')
    d = ImageDraw.Draw(ai)
    text(d,(48,96),'AIが行う作業',42)
    text(d,(48,180),'テーブル中央の黒い器を、皿の上へ',28,TEAL)
    text(d,(48,258),'入力：ことば、前方・手元画像、腕の状態',24)
    text(d,(48,310),'出力：ロボットを動かす予測動作列',24)
    text(d,(48,390),'モデル：公開済みSmolVLA',26)
    text(d,(48,442),f'実行：LIBERO-plus / MuJoCo / {data["guarded"]["device"].upper()}',24)
    text(d,(48,516),'既存のモデル・基本タスクを再利用',24,MUTED)
    ai.paste(Image.fromarray(normal[0]).resize((440,440)),(792,180))
    text(d,(48,634),'今回の追加：画像異常の監視、移動指令の停止、復旧後の再計画',24,TEAL)

    fault = base('WHAT DOES CAMERA LOSS MEAN?')
    d = ImageDraw.Draw(fault)
    text(d,(48,94),'AI入力用の手元画像だけを、全黒に置き換える',36)
    text(d,(48,156),f'{start:g}〜{end:g}秒の{end-start:g}秒間。前方カメラとシミュレータの照明は正常。',26,MUTED)
    for x,index,caption in [(80,max(facts['start']-1,0),'正常'),(512,middle,'画像欠損'),(944,facts['end'],'復旧')]:
        fault.paste(Image.fromarray(wrist[index]).resize((256,256),Image.Resampling.NEAREST),(x,240))
        text(d,(x,524),f'{caption} / {index/FPS:.2f}秒',24,TEAL if index!=middle else '#FF8996')
    text(d,(58,616),'比較映像の小窓には、この実際の入力画素を時刻を合わせて表示します。',24)

    mechanism = base('NEW STOP / RECOVERY CONTROL')
    d = ImageDraw.Draw(mechanism)
    text(d,(48,96),'今回追加した仕組み',42)
    for y,txt in [(200,'1. カメラ入力の欠落・不正値・全黒画像を検知'),(280,'2. 異常中は腕の移動指令を0にし、グリッパー指令を維持'),(360,'3. 正常画像が3フレーム続いたら、古い動作計画を破棄'),(440,'4. 現在の画像と腕の状態から、AIが動作を再予測')]:
        text(d,(58,y),txt,28,TEAL if y==280 else WHITE)
    text(d,(58,548),f'実測：{facts["pause"]/FPS:.2f}秒に停止、{end:.2f}秒に画像復旧、{facts["resume"]/FPS:.2f}秒に再開',26)
    text(d,(58,614),'異常検知と停止判断はルール処理。AIの追加学習はしていません。',24,MUTED)

    results = base('OBSERVED RESULTS / ONE INITIAL STATE')
    d = ImageDraw.Draw(results)
    text(d,(48,94),f'同じ初期配置・seed {facts["seed"]}での実測',36)
    table=[('条件','結果','シミュレーション時間')]
    for key,label in [('normal','画像正常（参考）'),('unprotected','画像欠損・対策なし'),('guarded','画像欠損・停止復旧あり')]:
        row=rows[key]
        table.append((label,outcome(row),f'{row["simulated_seconds"]:.2f} 秒'+('で打切り' if not row['success'] else '')))
    for i,cells in enumerate(table):
        y=190+i*78
        d.rounded_rectangle((48,y-8,1232,y+54),radius=5,fill='#1A2C3B' if i%2==0 else BG)
        for x,txt in zip([64,650,862],cells): text(d,(x,y),txt,24,TEAL if i==3 else WHITE)
    text(d,(58,544),f'停止中{facts["paused_steps"]}ステップ：腕6成分の指令がすべて0であることを確認。',26,TEAL)
    text(d,(58,612),'完了時間は制御ステップ数 ÷ 20。推論待ちを含む実時間ではありません。',22,MUTED)

    limits = base('SCOPE & NEXT STEPS')
    d = ImageDraw.Draw(limits)
    text(d,(48,96),'今回示せたことと、検証の範囲',36)
    for y,txt in [(200,'人工的な画像欠損を検知し、停止・再計画・タスク再開を実証。'),(280,'1タスク・1配置・各条件1回。一般的な成功率の改善は未確認。'),(360,'フリーズ画像など、すべての故障を検知する仕組みではない。'),(440,'実機での安全性や、物理的に完全停止することは未検証。')]:
        text(d,(58,y),txt,26,TEAL if y==200 else WHITE)
    text(d,(58,560),'今後：初期配置・異常時刻を増やし、復旧後の達成を確認する。',26,TEAL)

    credits = base('CREDITS & REPRODUCIBILITY')
    d = ImageDraw.Draw(credits)
    text(d,(48,102),'モデルを使う仕組みに、停止・復旧の工夫を加える',36)
    for y,txt in [(220,'モデル：huggingface.co/lerobot/smolvla_libero_plus'),(285,'環境：github.com/sylvestf/LIBERO-plus'),(350,'制御基盤：github.com/huggingface/lerobot')]: text(d,(58,y),txt,26)
    text(d,(58,460),'コード・入力画像・動作ログ・説明資料を保存。既存部分と追加部分を明記。',24,TEAL)
    text(d,(58,550),'映像はシミュレーション時間で等速。推論待ち時間は含めません。',24,MUTED)
    text(d,(58,600),'完了後の静止表示は画面内に明記しています。',24,MUTED)

    final=out/f'PAI最終課題_{args.account}.mp4'
    encoder=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(final)],stdin=subprocess.PIPE)
    frame_count=0

    def emit(image, seconds=1/FPS):
        nonlocal frame_count
        raw=image.tobytes()
        for _ in range(round(seconds*FPS)):
            encoder.stdin.write(raw)
            frame_count+=1

    def card(image, seconds, name):
        image.save(preview_dir/f'{name}.png')
        emit(image,seconds)

    try:
        card(intro,8,'01_intro')
        card(ai,8,'02_ai')
        for i in range(8*FPS):
            im=base('REFERENCE / CAMERA INPUT NORMAL')
            d=ImageDraw.Draw(im)
            text(d,(48,92),'参考：画像が正常なときの片付け',36)
            im.paste(Image.fromarray(normal[min(i,len(normal)-1)]),(64,144))
            text(d,(632,210),'固定の英語指示をAIへ入力',28,TEAL)
            text(d,(632,280),'器をつかみ、皿の上へ移動',28)
            text(d,(632,350),outcome_sentence(rows['normal']),28,TEAL)
            text(d,(632,450),f'初期配置{facts["initial_state_index"]} / seed {facts["seed"]}',24,MUTED)
            status=('完了・最終状態を静止表示' if rows['normal']['success'] else '時間上限・最終状態を静止表示') if i>=len(normal)-1 else f'{i/FPS:.2f} 秒 / AIの指令で実行中'
            text(d,(632,530),status,24)
            text(d,(64,678),'シミュレーション時間で等速 / 推論待ちを除く',20,MUTED)
            if i==40: im.save(preview_dir/'03_normal.png')
            emit(im)
        card(fault,7,'04_fault')
        for frame in compare: emit(Image.fromarray(frame))
        card(mechanism,10,'06_mechanism')
        card(results,8,'07_results')
        card(limits,7,'08_limits')
        card(credits,5,'09_credits')
    finally:
        encoder.stdin.close()
        if encoder.wait()!=0: raise RuntimeError('Encoding failed')
    duration=frame_count/FPS
    assert 60<=duration<=180
    make_pdf(args,data,rows,facts)
    description=(f'タイトル：PAI最終課題_{args.account}\n\n'
        '画像が消えたら止まり、戻ったら片付けを再開するロボット\n\n'
        '公開済みSmolVLAで器を皿へ移すロボットに、カメラ入力の異常監視・停止・復旧処理を追加しました。AIの追加学習はしていません。\n'
        f'手元カメラ画像を作業開始{start:g}〜{end:g}秒の間だけ人工的に全黒にし、同じ初期配置・seedで対策の有無を比較しました。前方カメラは正常です。\n'
        f'対策なしは{outcome_sentence(rows["unprotected"])}。停止・復旧処理ありは{recovery_sentence(facts,rows["guarded"])}\n'
        '小窓は制御へ渡した実際の手元画像です。AIが画像を参照するのは動作列を更新する再推論時です。異常検知と停止判断はルール処理です。\n'
        '1タスク・1配置・各条件1回の実証であり、一般的な成功率改善・実機安全性・全故障への対応は未検証です。\n\n'
        '映像はシミュレーション時間で等速（20 Hz）です。推論待ち時間を含まず、倍速編集はしていません。完了後の静止表示は画面内に明記しています。\n\n'
        'モデル：https://huggingface.co/lerobot/smolvla_libero_plus\n環境：https://github.com/sylvestf/LIBERO-plus\n'
        'コードURL：GitHubで公開後に追記\n')
    (out/'video_description.txt').write_text(description)
    package_source(out,args.input)
    info={'topic':'camera blackout stop and recovery','account':args.account,'video':final.name,'duration_seconds':duration,'fps':FPS,'resolution':[1280,720],'report':f'説明資料_PAI最終課題_{args.account}.pdf','source_zip':'pai-final-2026-source.zip','experiment':f'one task, one initial state, seed {facts["seed"]}, one rollout per condition','external_publication':{'github':False,'youtube':False,'omnicampus':False}}
    (out/'build_info.json').write_text(json.dumps(info,indent=2)+'\n')
    print('VIDEO',final,duration)
    print('REPORT',out/f'説明資料_PAI最終課題_{args.account}.pdf')


def make_pdf(args,data,rows,facts):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as PdfImage, PageBreak
    pdfmetrics.registerFont(TTFont('ReportJP',str(args.pdf_font)))
    body=ParagraphStyle('body',fontName='ReportJP',fontSize=9.2,leading=15,textColor=colors.HexColor('#273B49'),wordWrap='CJK',spaceAfter=8)
    head=ParagraphStyle('head',parent=body,fontSize=12,leading=18,textColor=colors.HexColor('#127765'),spaceBefore=9,spaceAfter=8)
    title=ParagraphStyle('title',parent=body,fontSize=18,leading=27,textColor=colors.HexColor('#142A3A'),spaceAfter=8)
    small=ParagraphStyle('small',parent=body,fontSize=8,leading=12)
    out=args.output
    path=out/f'説明資料_PAI最終課題_{args.account}.pdf'
    doc=SimpleDocTemplate(str(path),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=59,bottomMargin=43,title='カメラ異常で停止し、復旧後に片付けを再開するロボット',author='')
    story=[]
    def p(txt,style=body): story.append(Paragraph(txt,style))
    p('画像が消えたら止まり、<br/>戻ったら片付けを再開する',title)
    p('SmolVLAのロボット操作に、カメラ異常への停止・復旧処理を追加',head)
    p(f'Physical AI応用編1 最終課題 / OmniCampus：{args.account} / 2026年10月31日',small)
    p('1. 概要と目的',head)
    p('公開済みAIで「テーブル中央の黒い器を皿へ移す」ロボットに、カメラ入力が失われたときの停止・復旧処理を追加した。画像が不正なまま動作を続ける場合と、移動指令を止めて画像復旧後に再計画する場合を、同じ条件で比較する。')
    p('2. AIと既存環境の利用',head)
    p('Vision-Language-Actionモデル lerobot/smolvla_libero_plus に、固定の英語指示、前方・手元のRGB画像、手先の位置・姿勢・グリッパー状態を入力し、50ステップの動作列を予測させる。通常は25ステップずつ実行して再推論する。モデル・基本タスク・LIBERO-plus環境・コンペ用の推論アダプターを再利用し、本課題で追加学習は行っていない。')
    p('3. 今回追加した工夫',head)
    p('画像の欠落・形状不正・非有限値・全黒画像を毎制御ステップで監視する。異常中は腕6成分の移動指令を0にし、グリッパーは直前の指令を維持する。正常画像を3フレーム連続で受け取ったら、古い動作列を破棄し、現在の画像・腕の状態から新しい動作をAIに予測させる。異常の検知と停止判断は今回追加したルール処理である。')
    p('全黒への置換後の実際の手元画像を記録し、全体映像と同時刻の小窓で表示した。停止中はAIを呼び出さず、監視処理が復旧を確認する。AIが画像を使うのは動作列を再予測するときである。')
    story.append(PdfImage(str(out/'camera_blackout.png'),width=450,height=253.125))
    p(f'画像欠損中の{(facts["start"]+facts["end"])//2/FPS:.2f}秒時点。小窓は実際の全黒入力。左は動作を続け、右は腕の移動指令を停止。',small)
    story.append(PageBreak())
    p('4. 実験条件',head)
    p(f'対象はLIBEROの器を皿へ移す1タスク、保存初期状態index {facts["initial_state_index"]}、seed {facts["seed"]}。各条件を1回実行した。制御20 Hz、最大{data["guarded"]["max_steps"]}ステップ（{data["guarded"]["max_steps"]/FPS:g}秒）、入力画像128×128、全体映像512×512。開始前に10ステップ静定した。成功はLIBEROのタスク達成条件で判定し、コンペ独自の衝突条件は適用していない。')
    versions=data['guarded']['versions']
    p(f'異常条件は手元画像だけをstep {facts["start"]}〜{facts["end"]-1}（{facts["start"]/FPS:g}秒以上{facts["end"]/FPS:g}秒未満）に全黒へ置換した。前方カメラの設定・物理シーンの初期設定・モデル重み・初期状態・seedはそろえた。通常時は画像欠損のない参考試行。環境はLIBERO-plus 4976dc3、robosuite {versions["robosuite"]}、MuJoCo {versions["mujoco"]}、LeRobot {versions["lerobot"]}、PyTorch {versions["torch"]} / {data["guarded"]["device"].upper()}。')
    p('5. 実測結果',head)
    table=[['条件','タスク達成','ステップ数','シミュレーション時間']]
    for name,label in [('normal','画像正常（参考）'),('unprotected','画像欠損・対策なし'),('guarded','画像欠損・停止復旧あり')]:
        row=rows[name]
        table.append([label,'達成' if row['success'] else '未達成',str(row['steps']),f"{row['simulated_seconds']:.2f} 秒"+('で打切り' if not row['success'] else '')])
    t=Table(table,colWidths=[175,75,75,186],repeatRows=1)
    t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'ReportJP'),('FONTSIZE',(0,0),(-1,-1),8.3),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#14384B')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#EFF5F6'),colors.white]),('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),9),('ALIGN',(1,1),(-1,-1),'CENTER')]))
    story.append(t);story.append(Spacer(1,9))
    p('停止復旧ありでは'+recovery_sentence(facts,rows['guarded'])+f'停止中{facts["paused_steps"]}ステップで腕6成分の指令がすべて0、グリッパー指令が停止前と同じであることを保存ログで確認した。')
    p('時間は制御ステップ数÷20。動画はこのシミュレーション時間で等速再生し、推論待ちを含む実時間ではない。倍速編集は行っていない。',small)
    p('6. 考察と限界',head)
    p('この1例では、異常中に動作を進めず、復旧後の観測から再計画する処理によりタスクを再開できた。異なる初期配置や異常時刻での成功率改善・一般的な優位性は未確認。フリーズ画像など、全故障を検知する仕組みではない。腕への移動指令0は物理的な完全停止や実機の安全性の証明ではなく、衝突回避も検証していない。今後は配置・発生時刻・欠損時間を増やして検証する。')
    p('7. 出典・再現性',head)
    p('モデル：https://huggingface.co/lerobot/smolvla_libero_plus<br/>環境：https://github.com/sylvestf/LIBERO-plus<br/>実装基盤：https://github.com/huggingface/lerobot<br/>モデルrevision：7bb70aa5bc92b82c9239142775d3a173103567ff<br/>ソース一式に実行手順、条件別の結果・動作・入力画像ログを保存。重みと第三者アセットは提供元から取得する。',small)
    def page(canvas,doc):
        canvas.setFont('ReportJP',8);canvas.setFillColor(colors.HexColor('#416173'))
        canvas.drawString(42,A4[1]-32,'PAI FINAL PROJECT / CAMERA LOSS & RECOVERY')
        canvas.drawRightString(A4[0]-42,A4[1]-32,args.account)
        canvas.setStrokeColor(colors.HexColor('#C9D7DF'));canvas.line(42,A4[1]-41,A4[0]-42,A4[1]-41)
        canvas.drawRightString(A4[0]-42,26,str(doc.page))
    doc.build(story,onFirstPage=page,onLaterPages=page)


def package_source(out,source_input):
    names=['README.md','SUBMISSION.md','pyproject.toml','uv.lock','.gitignore','run_demo.py','offline_policy.py','camera_recovery.py','prepare_assets.py','build_camera_submission.py','experiments/probe_camera_recovery.py','experiments/build_camera_preview.py','experiments/README.md']
    files=[(ROOT/name,Path(name)) for name in names]
    for case in ['normal','unprotected','guarded']:
        files.extend((p,Path('results/camera-recovery')/case/p.name) for p in (source_input/case).iterdir() if p.is_file() and p.suffix in ['.json','.mp4','.npz'])
    for file,_ in files:
        if file.suffix in ['.md','.py','.toml','.json','.lock']:
            value=file.read_text()
            if str(Path.home()) in value: raise ValueError(f'Personal path in {file.name}')
    archive=out/'pai-final-2026-source.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for file,relative in files: z.write(file,Path('pai-final-2026')/relative)
    with ZipFile(archive) as z:
        assert z.testzip() is None
    print('SOURCE',archive,len(files),'files')


if __name__=='__main__':
    main()
