
import curses
from time import sleep
from turing.ss.simulator import Simulator


def map_action(action: str, stack: str):
    if action == "push0":
        stack, a, b = "(" + stack, "0", "1"
    elif action == "push1":
        stack, a, b = ")" + stack, "1", "1"
    elif action == "pop":
        stack, a, b = stack[1:] if len(stack)>0 else "", "1" if len(stack)>1 and stack[1]==")" else "0", "1" if len(stack)>1 else "0"
    else:
        stack, a, b = stack, "1" if len(stack)>0 and stack[0]==")" else "0", "1" if len(stack)>0 else "0"
    return stack, a, b


def render(stacks, z, attr):
    p = len(stacks)
    pad = 1
    stdscr.clear()
    for j, stack in enumerate(stacks):
        pos = pad
        l = f"stack {j}: "
        stdscr.addstr(pad+j, pos, l, curses.A_DIM)
              
        if len(stack) > 0:
            # highlight top
            pos += len(l)  
            l = stack[0]
            stdscr.addstr(pad+j, pos, l, attr | curses.A_BOLD)

        if len(stack) > 1:
            pos += len(l)
            l = f"{stack[1:]}"
            stdscr.addstr(pad+j, pos, l)
        
    # show machine attributes
    pos = pad
    l = "state: "
    stdscr.addstr(pad+p, pad, l, curses.A_DIM)
    pos += len(l)
    stdscr.addstr(pad+p, pos, z, curses.color_pair(2))
    stdscr.refresh()


tx = Simulator()
# string = "()((()(()))())"
string = "()"

try:    
    stdscr = curses.initscr()
    curses.curs_set(0)
    curses.start_color() 
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)

    sleep(2)
    z, a_1, a_2, b_1, b_2 = "I", "0" if string[0]=="(" else "1", "0", "1", "0"
    action_1, action_2 = None, None
    stack_1 = string
    stack_2 = ""
    for i in range(20):
        render([stack_1, stack_2], z, curses.color_pair(1))        
        sleep(1)
        # next step
        zpre = z
        z, action_1, action_2 = tx.full_description[(z, a_1, a_2, b_1, b_2)]
        if action_1 == action_2 == "pop":            
            render([stack_1, stack_2], zpre, curses.color_pair(3))
            sleep(1)
        stack_1, a_1, b_1 = map_action(action_1, stack_1)
        stack_2, a_2, b_2 = map_action(action_2, stack_2)

    render([stack_1, stack_2], z, curses.color_pair(1))
        
finally:
    curses.echo()
    curses.nocbreak()
    curses.endwin()