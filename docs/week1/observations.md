# Week 1 — First Logger & Observations

The logger was created and grapher was created, although primitive. One thing I noticed (although from eye, not very accurate), was that the data it self was cleaner than what I expected, meaning that there weren't a lot of stutters.

I initially thought that it may have been an issue with how MacOS handles pynput mouse, but timestamps seem to suggest that the on_move event was happening approximately every 0.001 seconds, which makes sense for a mouse with around 1000Hz polling rate.

This seems to suggest that mouse movement itself is actually more smoother than what one would think it is.

However, on the circle test, the aim seemed to be much more shakier than flicks. It seems that generally the slower the mouse movement is, the shakier the input is, meaning that it requires a lot more filtering.

This seems to suggest that as players become more advanced and their aim speed increases, maybe their aim becomes more efficient even in the way of moving.

However, this could also be from how the circle test was involving a lot of direction change.
