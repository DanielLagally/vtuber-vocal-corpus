# Frame-by-frame pitch, one value per line, 0 for unvoiced.
#
# Driven as a subprocess because the pip binding (praat-parselmouth) embeds a
# much older Praat than the standalone program, so the newer pitch methods --
# filtered autocorrelation in particular, which exists specifically to reduce
# octave errors -- are unreachable through it.
#
# Praat resolves relative paths against the SCRIPT's directory, not the working
# directory, so the caller must pass an absolute infile path.
#
# method is "filtered_ac", "ac" or "cc". For filtered autocorrelation the third
# argument is Praat's "pitch top" (where attenuation begins), not a hard
# ceiling -- the method is parameterised differently on purpose.

form: "Pitch"
    sentence: "infile", ""
    real: "floor", "60"
    real: "ceiling", "800"
    real: "hop", "0.02"
    sentence: "method", "filtered_ac"
endform

snd = Read from file: infile$

if method$ = "filtered_ac"
    To Pitch (filtered autocorrelation): hop, floor, ceiling, 15, "no", 0.03, 0.09, 0.50, 0.055, 0.35, 0.14
elsif method$ = "cc"
    To Pitch (cc): hop, floor, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, ceiling
else
    To Pitch (ac): hop, floor, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, ceiling
endif

n = Get number of frames
# First line is the centre time of frame 1, so the caller can evaluate other
# analyses at exactly these times and make voiced-gating exact rather than
# correct to within a frame of alignment slop.
t0 = 0
if n > 0
    t0 = Get time from frame number: 1
endif
appendInfoLine: "t0 ", t0
for i to n
    f = Get value in frame: i, "Hertz"
    if f = undefined
        appendInfoLine: 0
    else
        appendInfoLine: f
    endif
endfor
