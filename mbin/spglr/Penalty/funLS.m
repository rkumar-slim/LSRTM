function [f,g] = funLS(r, params)

if params.ls==1
    f = norm(r,2);
    g = r./f;
else
    f = 0.5*norm(r)^2;
    g = r;
    
end

end