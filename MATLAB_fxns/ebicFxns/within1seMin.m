function [val,ind] = within1seMin(data)
%WITHIN1SE get largest lambda within 1 standard error of min.
%   Note:assume data is sorted smallest to largest
    se = std(data)/sqrt(length(data));
    [minval, minind] = min(data);
    maxval = minval + se;
    ind = minind + sum(data((minind + 1):end) < maxval);
    val = data(ind);
end

