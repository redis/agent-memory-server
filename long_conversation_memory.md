# Story 1: Long Conversation Memory

## User story
As an agent, I can keep a long conversation in working memory and still get useful recent context after the session grows large.

## Expected functionality
- Long conversations are stored successfully with O(1) latency.
- Older content is summarized into context when needed and configured.
- Recent messages stay available and in order regardless of length.
- Reading the session or building a memory prompt still works after summarization, regardless of length.

## Why it matters
This is the basic "the agent remembers the conversation" experience.

## What we expect to break
- Recent messages getting lost.
- Messages coming back in the wrong order.
- Summaries not appearing, or being empty.
- Session reads becoming inconsistent after many updates.

## Pass criteria
- Recent turns are still there.
- Summary appears when the session gets large.
- The session is still readable and useful afterward.
- How to test it

## Prepare:
- one short conversation
- one medium conversation
- one very long conversation
- one conversation with a few very large messages

## Run:
- repeated updates to one session
- many separate long sessions in parallel
- concurrent updates to the same session

## Check:
- was a summary created?
- are the last few messages still present?
- are the messages in the right order?
- does prompt generation still include the expected context?

## Follow up questions

- how do we define small/medium/large?
- every k token it kicks off the summary mechanism
- consider planning multiple different trips with the same user
- switch conversation in a single thread
