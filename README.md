Reflections
Week 0 
I initially had an idea on how something equivalent to rapid-trigger would look like for the movement of the mouse. I thought that since our input that we want is equivalent to the direction of the force we are applying, we could perhaps get the change in velocity of the mouse instead as a source of input instead of the displacement of the mouse. 

While thinking about it, I remembered other sources of error in our aim. For example, our aim moves on a curve instead (source needed, off from memory), which is why it is not exactly a straight line. Not only that, but theres the irony that if you want to get your aim as fast as possible to a position, you will overaim, since you are not able to slow down. However, if you want to aim exactly at that possition, you would need to slow down your mouse way before you reach the point, which is a waste of time. 

The way we move our mouse is also obviously very important to the resulting aim. For example, if we lock our shoulders and elbow and only move our forearm (meaning our elbow is a pivot), our mouse position will effectively move like a circle. Even if we only move our wrist, the mouse position will move like a circle, since our wrist is a pivot. 

The way we move our mouse anatomically seems to be of multiple pivots. 

Week 1

The logger was created and grapher was created, although primitive. One thing I noticed (although from eye, not very accurate), was that the data it self was cleaner than what I expected, meaning that there weren't a lot of stutters. 

I initially thought that it may have been an issue with how MacOS handles pynput mouse, but timestamps seem to suggest that the on_move event was happening approximately every 0.001 seconds, which makes sense for a mouse with around 1000Hz polling rate. 

This seems to suggest that mouse movement itself is actually more smoother than what one would think it is. 

However, on the circle test, the aim seemed to be much more shakier than flicks. It seems that generally the slower the mouse movement is, the shakier the input is, meaning that it requires a lot more filtering. 

This seems to suggest that as players become more advanced and their aim speed increases, maybe their aim becomes more efficient even in the way of moving.

However, this could also be from how the circle test was involving a lot of direction change. 

Week 2

