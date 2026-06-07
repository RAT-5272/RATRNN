### Where is the data?

I included the training data files in .gitignore because I don't want to put 100+ MB (at the time) of my own personal conversations online. You can use your own I guess!

### Format

The format used for training data is quite simple and also quite bad, I made it at the start of the whole project and it made the parsing unnecessarily difficult, I should have added separating characters between the things in the header, and I also should have at least used JSON for storing instead of plain text. It isn't that bad though because it still has more than enough information. Too late to change it to a better format now though.

It strictly follows this format, if there is no message being replied to the MessageReplyID = 0, and MsgReplyStart-End = "None".

<BOT_MsgHeadStart>MessageID UserID Username DisplayName Timestamp MessageReplyID<BOT_MsgHeadEnd>
<BOT_MsgReplyStart>Content of the message that a user is replying to<BOT_MsgReplyEnd>
<BOT_MsgContentStart>Content of the message that a user sent<BOT_MsgContentEnd>