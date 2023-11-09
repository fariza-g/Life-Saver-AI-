# Life-Saver-AI-
Lifesaver AI is an innovative solution aimed at protecting people's lives and preventing tragic events. These are the project device program codes.

main.py is designed to detect a person on the roof and send a message 5 seconds after that.
Also, after detection, the program sends command '1' to Arduino in order to trigger the speaker to produce sound.

finalCode.py includes person detection, sending SMS, and sending a link to a live stream so that security authorities are aware of what is happening on the roof of the building.
Also, the program sends a command to the Arduino to trigger the speaker.

stream.py is the code for a separate live stream, we used Flask to implement it.
It uses a template index.html that is created for a web page and then sends an SMS to the police.
