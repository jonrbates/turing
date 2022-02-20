import curses
from time import sleep
from turing.wcm.simulator import Simulator


def render(tape, state, head):
    cursor = "^"
    pad = 1
    stdscr.clear()
    # show tape
    pos = pad
    l = "tape:  "
    stdscr.addstr(pad, pos, l, curses.A_DIM)
    pos += len(l)
    l = tape
    stdscr.addstr(pad, pos, l)
    # show head
    pos = pad
    l = "head:  "
    stdscr.addstr(pad+1, pos, l, curses.A_DIM)
    pos += len(l) + head
    l = cursor
    stdscr.addstr(pad+1, pos, l, curses.color_pair(1) | curses.A_BOLD)
    # show state
    pos = pad
    l = "state: "
    stdscr.addstr(pad+2, pos, l, curses.A_DIM)    
    pos += len(l)
    l = state
    stdscr.addstr(pad+2, pos, state, curses.color_pair(2))    
    
    stdscr.refresh()


tx = Simulator()
delta = tx.delta
tape = "B()((()(()))())E"
head = 0    
state = "I"
n = len(tape)

final_states = ["T", "F"]


try:
    stdscr = curses.initscr()
    curses.curs_set(0)
    curses.start_color() 
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)

    sleep(3)
    while state not in final_states:  
        render(tape, state, head)
        # update
        (state, write, move) = delta[(state, tape[head])]        
        # write
        tape = tape[:head] + write + tape[head+1:]
        # move
        head += move
        sleep(.3)
    render(tape, state, head)
    sleep(.3)

finally:
    curses.echo()
    curses.nocbreak()
    curses.endwin()