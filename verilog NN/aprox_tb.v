module aprox_tb();

    reg signed [31:0] b= 1;
    wire signed [31:0] out_val;
    initial
    begin
        #15;
        b= 65534; // from 0-1
        #15;
        b= 65536;//from 1.000001 -2
        #15
        b= 131072;
        #15;
        b= 131073;//from 2.0000152587890625
        #15;
        b= 2147418112;
      
    end

    aprox a(out_val, b);

    initial
     $monitor("At time %t, out:%f b:%f ", $time, out_val, b);
endmodule