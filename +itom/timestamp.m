function ts = timestamp()
	% "Y-M-D-h-m"
    t = datetime();
    y = num2str(year(t));
    M = num2str(month(t));
    d = num2str(day(t));
    h = num2str(hour(t));
    m = num2str(minute(t));
    ts = strcat(y, "-", M, "-", d, "-", h, "-", m);
end
