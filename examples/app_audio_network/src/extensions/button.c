#include <xs1.h>
#include <xcore/chanend.h>
#include <xcore/port.h>

void button_server(chanend_t c_buttons) {
    port_t buttons = XS1_PORT_4E;
    int value = 0;
    port_enable(buttons);
    while(1) {
        port_set_trigger_in_not_equal(buttons, value);
        value = port_in(buttons);
        chanend_out_word(c_buttons, value);
        chanend_out_end_token(c_buttons);
        chanend_check_end_token(c_buttons);
    }
}
