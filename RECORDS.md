# multi-label - 1 loss
Q1. is out_scale useful? If so, what is the best init technique? (moving from no scale to scale with init factors 1, 0.1, 0.01, 0...)
   
Result: not useful here.
   
Performance: no scale > 1 > 0.1 > 0.01 > 0
   
Report: R1

---

Q2. Does reducing T result in better learning through dense representation? (moving from crnn-1 to crnn-2)
   
Result: No
   
Performance:
      
- crnn-1 (big T) is slightly better that crnn-2 fpr e-2, e-3 & e-4 (small T)
- crnn-2 (small T) is slightly better that crnn-1 fpr e-0, e-1 (small T)
   
Report: R2

---

Q3. Which is better? Unicode or multi-label without additional losses?
   
Close fight
   
Multi-label config: no scale, crnn-1/2
   
Report: R2

---

Q4. Does softmaxing consonant_out and glyph_out help?
   
No

---

# multi-label - 3 losses
Q5. What weight split & out_scale combo useful? If so, what is the best  technique?
   Benchmarks: (unicode, crnn-1), (multi-label, crnn-1, use_out_scale=False) 
   
   ## without out_scale:
      ### char-0.33, cons-0.33, gly-0.33
      bad
      ### char-0.5, cons-0.25, gly-0.25
      bad
      ### char-0.7, cons-0.15, gly-0.15
      bad
      ### char-0.7, cons-0.3, gly-0
      good
      ### char-0.7, cons-0, gly-0.3
      good++
   
Result: Learning from 2 losses seems to be helping over learning from 3 losses. Performance is close to learning without additional losses
   
Report: R3
   
   ## with out_scale:
      ### char-0.7, cons-0.15, gly-0.15
      good
      ### char-0.7, cons-0.3, gly-0
      bad
      ### char-0.7, cons-0, gly-0.3
      good
   
Result: Over fitting, Achieved zero train loss
   
Suggestions: Train more data, increase weight decay, add BN
   
Report: R4

   ## with single out_scale:
      ### char-0.7, cons-0.15, gly-0.15
      good+
      ### char-0.7, cons-0.3, gly-0
      good
      ### char-0.7, cons-0, gly-0.3
      good
   
Result: Over fitting, Achieved zero train loss
   
Suggestions: Train more data, increase weight decay, add BN
   
Report: R5

>  Training with character loss(0.7) and glyph loss(0.3) without out_scale is slightly better than other experiments...

>  When training with out_scale, if I just reduce the glyph loss, the consonant loss also gets reduced and vice versa 

>  out_scale helps to reduce cons/glyph loss while learning from 2 other losses
---

Q6. Does softmaxing consonant_out and glyph_out help?

   
   ## without out_scale:
      ### char-0.7, cons-0.15, gly-0.15
      bad
      ### char-0.7, cons-0.3, gly-0
      bad
      ### char-0.7, cons-0, gly-0.3
      bad

Result: Bad Results

Suggestions: Don't use softmax without scale_out

Report: R6
   

   ## with out_scale:
      ### char-0.7, cons-0.15, gly-0.15
      bad
      ### char-0.7, cons-0.3, gly-0
      very bad
      ### char-0.7, cons-0, gly-0.3
      very bad

Result:  Bad Results

Suggestions: Don't use softmax at all

Report: R7


# Model Comparision