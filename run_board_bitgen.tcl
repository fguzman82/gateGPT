# Open the regenerated board project and run the full flow to a .bit
project open microgpt_fpga_board.xise
puts "=== running Generate Programming File ==="
if {[catch {process run "Generate Programming File"} res]} {
    puts "PROCESS ERROR: $res"
}
project close
puts "=== DONE ==="
