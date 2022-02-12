import curses
from time import sleep
from turing.translators import Translator

stdscr = curses.initscr()
curses.noecho()
curses.cbreak()

tx = Translator()
delta = tx.delta
tape = "B()((()(()))())E"
head = 0    
state = "I"
n = len(tape)
cursor = "^"
pad = " "
final_states = ["T", "F"]

try:
    sleep(2)
    while state not in final_states:  
        stdscr.addstr(0, 0, f"  {tape}")
        stdscr.addstr(1, 0, f"{state} {pad*head}{cursor}{pad*(n-head)}")
        stdscr.addstr(2, 0, f"")
        stdscr.refresh()
        # update
        (state, write, move) = delta[(state, tape[head])]        
        # write
        tape = tape[:head] + write + tape[head+1:]
        # move
        head += move
        sleep(.3)
finally:
    curses.echo()
    curses.nocbreak()
    curses.endwin()



