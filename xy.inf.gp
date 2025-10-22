set terminal pdf
set termoption noenhanced
set output "xy_grid1.inf.pdf"

do for [m in "u_norm u_prod uv_norm x_scale xy_scale"] {
    set title m
    plot "xy_grid1_".m.".train.tsv" t "train", "xy_grid1_".m.".inf.tsv" t "inf"
}