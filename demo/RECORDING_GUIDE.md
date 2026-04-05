# Demo Recording Guide

## What to record

Record this script on your phone (Voice Memos app). Don't try to sound polished — the whole point is that the raw recording sounds rough and phonepod makes it broadcast-ready.

## Setup (maximize the "before" contrast)

- Use your phone's built-in mic (not AirPods or external mic)
- Hold the phone at a natural distance (not pressed against your mouth)
- Record in a normal room — not a treated studio
- Leave background noise in: fan, traffic, keyboard, whatever's there
- Don't worry about mouth sounds, breaths, or plosives

## The Script

> Record this naturally. Don't read it word-for-word — just hit the key points in your own voice. Pause where it feels right. The chapter map below is a guide, not a teleprompter.

---

### Chapter 1: The problem (15-20 seconds)

Key points to hit:
- You record voice memos on your phone all the time
- They sound fine when you're listening back on the phone
- But the moment you try to put them in a podcast, or a video, or even a voice message to someone important — they sound terrible
- There's this gap between what your voice actually sounds like and what the phone captures

### Chapter 2: What exists today (10-15 seconds)

Key points to hit:
- There are AI models that can denoise audio — research demos, mostly
- There are professional mastering chains that podcasters use
- But nothing combines both into one tool that just works
- And almost everything requires uploading your audio to someone's server

### Chapter 3: What phonepod does (15-20 seconds)

Key points to hit:
- phonepod is a single command: you give it a phone recording, it gives you back podcast-quality audio
- Five stages: noise suppression, speech enhancement, EQ and compression, loudness normalization, limiting
- Everything runs locally on your machine — nothing leaves your computer
- Takes about 7 seconds for a 2-minute recording

### Chapter 4: The kicker (5-10 seconds)

Key points to hit:
- This audio you're hearing right now? It was recorded on a phone
- Then I ran `phonepod` on it
- That's it

---

## Total length: 45-65 seconds

## After recording

Save the voice memo as `demo_raw.m4a`, then:

```bash
# Copy to the project
cp demo_raw.m4a demo/before.m4a

# Run phonepod
phonepod demo/before.m4a demo/after.wav

# Listen to both
open demo/before.m4a demo/after.wav
```

The before/after pair goes in the README and the GitHub repo.
