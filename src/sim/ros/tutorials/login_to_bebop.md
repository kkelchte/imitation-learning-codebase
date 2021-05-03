Start the bebop by clicking the on-bottom once.
After the bebop is properly running, namely fan started second time, click the on-bottom 4x.
Connect to the bebop's wifi.
Open a terminal on your machine and login with telnet:
telnet 192.168.42.1

To make permanent changes, you have to make the file system writable:
mount -o remount,rw /

Check the PID of the dragon-prog to kill it with top or ps:
ps -ef | grep dragon
kill -9 <fill in PID>
restart dragon-prog with new settings, see -h for more option:
/usr/bin/dragon-prog -S 0 -s 640x480 -V 115
-S 0 disables video stabilization
-s 640x480 reduces the video streaming to lower resolution
-V 115 sets the field of view wider 

Good references are:
https://wiki.paparazziuav.org/wiki/Bebop
https://visp-doc.inria.fr/doxygen/visp-daily/tutorial-bebop2-vs.html
http://wiki.yobi.be/wiki/Parrot_Bebop
https://wiki.paparazziuav.org/wiki/Bebop


CHANGE LOG

** White bebop **
Changed /data/dragon.conf autorecord -> false.
Added default restart of dragon-prog in a startup_script.sh called from /etc/init.d/rcS 
Restart dragon prog with following settings:

 - S 0: video stabilisation off 
 - s 640x480: streaming resolution
 - f 24: frame rate
 - H 1: stream mode with low latency (slightly more blur)
 
 Unfortunately no value was found to set the field-of-view. This used to be -V. 