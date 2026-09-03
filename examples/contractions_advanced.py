#!/usr/bin/env python3
"""
Advanced English contractions with extensive direct speech using pykokoro.

This example features a narrative-heavy text with lots of dialogue and direct speech,
testing how contractions are pronounced in conversational contexts including:
- Negative contractions in dialogue (don't, can't, won't, shouldn't)
- Multiple contractions in speech (I'd've, shouldn't've, could've)
- Informal speech patterns (gonna, wanna, gotta)
- Contractions with quotation marks and speech attribution
- Natural conversational flow with mixed contraction types

Usage:
    python examples/contractions_advanced.py

Output:
    contractions_advanced_demo.wav - Generated speech with dialogue-heavy text
"""

import argparse
import logging

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.debug.segment_invariants import check_segment_invariants
from pykokoro.generation_config import GenerationConfig
from pykokoro.stages.audio_generation.noop import NoopAudioGenerationAdapter
from pykokoro.stages.audio_postprocessing.noop import NoopAudioPostprocessingAdapter
from pykokoro.stages.doc_parsers.ssmd import SsmdDocumentParser
from pykokoro.stages.g2p.noop import NoopG2PAdapter
from pykokoro.types import Segment, Trace

# Extensive text with lots of direct speech and contractions
TEXT = """
#The Conversation

##Chapter One: The Meeting

'I don't like you,' a man told me once, standing in the doorway of an old café.
His words hung in the air like smoke...

"I can't... or shouldn't," I replied, confused by his hostility. "We've never even met before!"

"That's exactly the point!" he said, leaning against the frame. "You're the type who wouldn't
understand. You'll never get it, no matter how hard you'll try!"

I felt my face flush. "I'll have you know that's completely unfair! You don't know anything about me!"

"Don't I?" He smirked. "I'd've thought you'd've figured it out by now... People like you—you're
all the same. You won't listen, you can't comprehend, and you shouldn't even bother trying!"

"That's ridiculous!" I protested. "I'm not gonna stand here and let you insult me! What's your
problem anyway?"

He laughed bitterly. 'What's my problem? You're asking what's my problem? I'll tell you what's
wrong—it's people who think they'll solve everything with empty words. They're gonna promise
the world, but they won't deliver... They'll say "I'd've helped if I could've," but that's just
an excuse, isn't it?'

## Chapter Two: The Argument

"Look..." I said, trying to stay calm, "I don't know what happened to you, but you shouldn't
take it out on strangers! That's not fair, and it won't make you feel better."

"Won't it?" he challenged. "You're telling me what'll make me feel better? That's rich,
coming from someone who doesn't know me!"

"You're right—I don't know you!" I admitted. "But I'd've listened if you'd've given me a chance...
I would've understood if you'd've explained. But you're not gonna do that, are you?"

He shook his head. 'Why should I? You'll just say what everyone says: "It's gonna be okay!"
or "You've gotta stay positive!" or "Things'll get better!" But they won't, and you can't
promise that they will!'

"I wasn't gonna say that..." I said quietly. "I was gonna say that I'm sorry you're hurting."

That stopped him. His expression softened slightly. "You're... what?"

"I'm sorry..." I repeated. 'I don't know what happened, and I won't pretend I do. But I can
see you're in pain, and that's real. Your feelings're valid, even if I don't understand them yet.'

## Chapter Three: The Story

He was silent for a long moment. Then he sighed. 'You wanna know what happened? You really
wanna hear this?'

"I'll listen..." I said. 'If you'll tell me.'

He looked away. "It's not something I'd've normally talked about... But... alright.
You've asked, so I'll tell you."

'Three years ago,' he began, "I'd've done anything for my family! I'd've given everything—
and I did. I worked two jobs, sometimes three. My wife'd say, 'You've gotta rest!' but I
couldn't. We'd bills to pay, mouths to feed..."

"That must've been exhausting..." I said.

'It was!' he continued. "But I'd've kept going. I wouldn't've stopped. I shouldn't've stopped.
But then... then it all fell apart..."

'What happened?'

"My company went under. Just like that! They'd promised we'd be fine, that they'll take
care of us. But they didn't! They couldn't, or wouldn't—I don't know which. Suddenly,
I'd lost everything I'd've worked for..."

## Chapter Four: The Revelation

'My wife said, "Don't worry, we'll manage." But I knew we wouldn't... Not really. The savings'd
run out, the bills'd pile up. I told her, "I can't do this!" and she said, "Yes, you can!
You've gotta try!"'

"But you couldn't've known it'd turn out badly..." I offered.

'Couldn't I?' he asked bitterly. "I should've seen the signs! They were there. I'd've noticed
if I'd've been paying attention. But I wasn't. I was too busy believing everything'd work out..."

'You can't blame yourself for that!' I said. "Nobody could've predicted—"

'Don't!' he interrupted. "Don't tell me I shouldn't blame myself! I'd've prevented all of
it if I'd've been smarter. My kids—they'd've had a better life if I'd've made different choices..."

'Where's your family now?' I asked gently.

He looked at the ground. "My wife... she'd said she'll stick by me. She'd promised she won't
leave. But she did... She'd've stayed, she said, if I'd've gotten help. But I wouldn't. I
couldn't admit I'd failed..."

## Chapter Five: The Understanding

'I'm so sorry...' I said. "That's... that's terrible..."

'Don't feel sorry for me!' he said harshly. "I don't deserve it! I'd've had everything, but
I threw it away... I wouldn't listen when people'd try to help. I can't forgive myself, and I
shouldn't expect others to!"

'But you've gotta forgive yourself eventually!' I said. "You can't carry this forever!"

'Can't I?' he asked. "You'd've thought the same thing... You'd've made the same mistakes.
Everyone says they won't, but they would! They'd've done exactly what I did!"

'Maybe...' I conceded. "Or maybe they'd've done worse. We're all human. We all make mistakes.
That's not an excuse, but it's the truth."

He looked at me for a long time. 'You're not what I'd've expected...'

"What'd you expect?"

'I'd've thought you'd judge me!' he said. "I'd've assumed you'll tell me what I should've
done. But you won't, will you?"

'No...' I said. "I won't. Because I don't know what I'd've done in your situation. Nobody does."

## Chapter Six: The Resolution

'You know what the worst part is?' he asked. "It's knowing that if I'd've had one more chance,
I'd've done it all differently! I wouldn't've made the same mistakes. I'd've listened to my
wife. I'd've asked for help... I'd've been better..."

'You still can!' I said. "It's not too late. You'll find a way forward if you're willing to try!"

'You think so?'

"I know so! You've already taken the first step—you've told your story. That's something you
said you wouldn't've done. But you did! That takes courage!"

He smiled weakly. 'I shouldn't've been so rude to you earlier... That wasn't fair. You didn't
deserve that.'

"It's alright..." I said. 'You were hurting. I understand.'

'You're kinder than I'd've been in your position!' he admitted. "I'd've walked away. I
wouldn't've stayed to listen..."

"But you would've..." I said. 'If you'd've met someone like yourself, someone hurting and
angry, you'd've stopped. You'd've listened. Because that's who you are.'

## Chapter Seven: New Beginnings

'How'd you know that?' he asked.

"Because you're standing here talking to me!" I said. 'You could've walked away. You'd've
been justified in doing that. But you didn't. You stayed. You shared your story... That tells
me everything I need to know.'

'I suppose you're right...' he said thoughtfully. "I'd've never thought of it that way."

'So what'll you do now?' I asked.

He thought for a moment. "I'm gonna try!" he said finally. 'I'm gonna reach out to my wife.
I don't know if she'll talk to me, but I've gotta try! I should've done it months ago...'

'That's a good start!' I encouraged him.

"And I'm gonna get help!" he continued. 'Real help. I should've done that from the beginning.
I'd've avoided so much pain if I'd've just admitted I needed support...'

'It's not easy admitting we need help...' I said. "But you're doing it now. That's what matters!"

'You know...' he said, "when I first saw you, I'd've sworn you'd be like everyone else. I
thought you'll judge me, that you won't understand. But I was wrong! I shouldn't've assumed that."

"We all make assumptions..." I said. 'The important thing is recognizing when we're wrong.'

## Chapter Eight: Parting Words

'I'd've never imagined this conversation'd go this way...' he said. "I thought I'd drive you
off with my anger. That's what I'd've wanted, I think. But you wouldn't go... You stayed."

'I'm glad I did!' I said sincerely.

"Me too..." he agreed. 'You've given me hope! I'd've thought I'd lost that forever, but maybe
I haven't... Maybe I'll find my way back.'

'You will!' I assured him. "It won't be easy, and it'll take time. But you'll get there!"

'I'd've liked to've met you sooner...' he said. "Maybe things'd've been different..."

"Maybe..." I said. 'Or maybe you needed to go through this to get to where you are now. We
can't know what would've happened. We can only know what will happen if we choose to move forward.'

'That's wise...' he said. "You're younger than me, but you've got wisdom I don't have!"

'I've got different experiences...' I said. "That's all. You've got your own wisdom that I don't
have. We've all got something to teach each other."

## Chapter Nine: The Promise

"Before you go..." he said, 'I wanna make a promise. I'm gonna do better! I won't give up, even
when it's hard. I'll keep trying, even when I'd rather quit. I'd've made this promise before,
but this time I'll keep it!'

'I believe you!' I said. "And if you ever need someone to talk to, you'll find me here. I
come to this café every Tuesday."

'Every Tuesday?' he asked.

"Every Tuesday!" I confirmed. 'Rain or shine. I'll be here if you need me.'

'I might take you up on that...' he said. "I'd've never thought I'd want to, but... I think
I will. Thank you..."

'You're welcome!' I said. "Take care of yourself."

'I'll try...' he said. "I really will. I won't make the same mistakes again. I can't promise
I won't make new ones, but I'll try to do better!"

"That's all anyone can do!" I said.

## Chapter Ten: Reflection

As I walked away from the café that day, I couldn't help but think about our conversation...
He'd've seemed so angry at first, so closed off. I'd've never guessed we'd end up having such
a meaningful exchange! But we did, and I'm glad we did.

I thought about what he'd said: 'I don't like you...' Those words'd hurt initially, but now I
understood they weren't really about me. They'd've been about his pain, his frustration, his
feeling of helplessness... He'd've lashed out at anyone who'd've been there.

But I'd stayed... I'd've left—many people would've. But something told me I shouldn't, that I'd
regret it if I did. And I'd've been right to stay!

I hoped he'd follow through on his promises. I hoped he'll contact his wife, that he'll get
help, that he'll find his way forward... I didn't know if he would, but I'd've done what I could.
I'd've listened, I'd've shown compassion, and I'd've given him hope.

That's all I could've done! That's all any of us can do—be present for others, listen without
judgment, and offer hope when it's needed most.

I'll be at that café next Tuesday. I don't know if he'll show up, but I'll be there just in
case... Because I'd've promised, and I won't break that promise. He'd've shown up if our
positions'd been reversed, I think. At least, I'd like to believe he would've...

And maybe, just maybe, that's enough! Maybe it's enough to've been there when someone needed
it. Maybe it's enough to've offered kindness when someone expected judgment... Maybe it's enough
to've listened when someone needed to be heard.

I'd've never expected this day to turn out the way it did! But I'm glad it did. I wouldn't've
changed it, even if I could've. Because sometimes, the most unexpected conversations're the
ones that matter most!

## Epilogue

The following Tuesday, I returned to the café as promised. I didn't know if he'd come, but
I'd've been there anyway... I'd've kept my word.

And he did come! He looked different—lighter somehow, as if a weight'd been lifted from his
shoulders. He'd've smiled when he saw me, a real smile this time!

'I did it!' he said simply. "I called her. We're gonna talk... Really talk. It won't be easy,
but we're gonna try!"

'That's wonderful!' I said, genuinely happy for him.

"I'd've never done it without you..." he said. 'You'd've given me the push I needed. Thank you...'

'You did it yourself!' I corrected gently. "I'd've just listened. You'd've made the choice
to change."

'Maybe...' he said. "But you'd've been there when I needed someone. That's what mattered!"

We talked for an hour that day, and many Tuesdays after. He'd've never expected to find a
friend that day, but he did! And I'd've never expected to make such a difference, but I had...

Sometimes, all it takes is listening... Sometimes, all it takes is showing up. Sometimes, all
it takes is saying, 'I'll be here!' and meaning it.

Because you'd've done the same! Anyone would've, if they'd've known how much it'd matter...

That's the power of kindness! That's the power of compassion! That's the power of simply
being there when someone needs you most!

And that's a power we've all got, if we'll only use it!

## The End
"""

VOICE = "af_bella"  # American Female voice for narrative
LANG = "en-us"  # American English


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce duplicate-word audio with full segment tracing."
    )
    parser.add_argument("--voice", default=VOICE, help="Voice name to use.")
    parser.add_argument("--lang", default=LANG, help="Language code.")
    parser.add_argument(
        "--pause-mode",
        default="tts",
        choices=("tts", "manual"),
        help="Pause handling mode.",
    )
    parser.add_argument(
        "--noop-g2p",
        action="store_true",
        help="Replace g2p with a no-op adapter (forces no-op synth).",
    )
    parser.add_argument(
        "--noop-synth",
        action="store_true",
        help="Replace synth with silence output.",
    )
    return parser.parse_args()


def print_segments(segments: list[Segment]) -> None:
    print("Segments:")
    for seg in segments:
        print(f"  {seg.id}: {seg.char_start}:{seg.char_end} text={seg.text!r}")


def print_phoneme_segments(phoneme_segments: list) -> None:
    print("Phoneme Segments:")
    for seg in phoneme_segments:
        print(f"  {seg.char_start}:{seg.char_end} text={seg.text!r}")
        print(seg)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s [%(name)s] %(message)s",
    )

    generation = GenerationConfig(lang=args.lang, pause_mode=args.pause_mode, pause_paragraph=0.9)
    cfg = PipelineConfig(voice=args.voice, generation=generation, return_trace=True)

    doc_parser = SsmdDocumentParser()

    noop_synth = args.noop_synth
    if args.noop_g2p and not args.noop_synth:
        print("Forcing no-op synth because no-op g2p omits tokens.")
        noop_synth = True

    g2p = NoopG2PAdapter() if args.noop_g2p else None
    audio_generation = NoopAudioGenerationAdapter() if noop_synth else None
    audio_postprocessing = NoopAudioPostprocessingAdapter() if noop_synth else None

    pipeline = KokoroPipeline(
        cfg,
        doc_parser=doc_parser,
        g2p=g2p,
        audio_generation=audio_generation,
        audio_postprocessing=audio_postprocessing,
    )

    print("=" * 70)
    print("ADVANCED CONTRACTIONS WITH DIRECT SPEECH")
    print("=" * 70)
    print("\nThis example features an extensive narrative with dialogue testing:")
    print("  - Common contractions in speech: don't, can't, won't, I'll, you'll")
    print("  - Negative contractions: shouldn't, couldn't, wouldn't, hasn't")
    print("  - Complex contractions: I'd've, you'd've, wouldn't've, couldn't've")
    print("  - Informal speech: gonna, wanna, gotta")
    print("  - Contractions with quotation marks and dialogue attribution")
    print("  - Natural conversational flow with mixed contraction patterns")
    print("  - Possessive forms: it's vs its, who's vs whose, you're vs your")
    print("=" * 70)
    print(f"\nVoice: {VOICE}")
    print(f"Language: {LANG}")
    print(f"\nText length: ~{len(TEXT)} characters")
    print("Estimated duration: ~15-20 minutes")

    print("\nGenerating audio (this may take a while for long text)...")

    output_file = artifact_path("contractions_advanced_demo.wav")
    result = pipeline.run(TEXT)
    result.save_wav(output_file)

    doc = doc_parser.parse(TEXT, cfg, Trace())

    print(f"clean_text length: {len(doc.clean_text)}")
    print_segments(result.segments)
    print_phoneme_segments(result.phoneme_segments)
    check_segment_invariants(result.segments, doc.clean_text)

    if result.trace and result.trace.warnings:
        print("Warnings:")
        for warning in result.trace.warnings:
            print(f"  - {warning}")

    print(f"Wrote WAV to: {output_file}")
    duration = len(result.audio) / result.sample_rate
    print(f"\nCreated {output_file}")
    print(f"Duration: {duration:.2f} seconds")
    print("\nNote: This is a long narrative with extensive dialogue.")
    print("Listen to verify correct pronunciation of contractions in context:")
    print("  - Dialogue: 'I don't like you,' 'I can't or shouldn't'")
    print("  - Complex: I'd've, you'd've, wouldn't've, shouldn't've")
    print("  - Natural speech patterns with mixed contractions")
    print("  - Contractions with quotation marks and attribution")


if __name__ == "__main__":
    main()
